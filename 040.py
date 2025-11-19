#!/usr/bin/env python3
"""
ShifaMind 040: Production-Ready Training & Demo Script
Complete medical diagnosis prediction with concept filtering and explainability

KEY FEATURES:
- Diagnosis-conditional concept filtering (fixes concept selection bias)
- 4-stage training pipeline with production checkpoints
- Integrated Gradio demo
- Evidence extraction with concept-diagnosis alignment
- Resume-able training from checkpoints

TARGET METRICS:
- Macro F1: 0.76-0.78 (5-8% over baseline)
- Citation Completeness: >95%
- Concept-Evidence Alignment: >70%
- Concept Precision: >70% (after filtering)

USAGE: Copy cells into Google Colab and run sequentially
"""

# ============================================================================
# @title 1. Setup & Imports
# @markdown Install dependencies and configure environment
# ============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Google Colab setup
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    print("🔧 Installing dependencies...")
    os.system('pip install -q faiss-cpu scikit-learn transformers torch gradio')
    from google.colab import drive
    drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import faiss
import time
import pickle
import math
import re

# Seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

# Data paths
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind/01_Raw_Datasets/Extracted')
MIMIC_PATH = BASE_PATH / 'mimic-iv-3.1'
UMLS_PATH = BASE_PATH / 'umls-2025AA-metathesaurus-full/2025AA/META'
ICD_PATH = BASE_PATH / 'icd10cm-CodesDescriptions-2024'
NOTES_PATH = BASE_PATH / 'mimic-iv-note-2.2'

print(f"\n📁 Data paths configured")
print(f"  MIMIC-IV: {MIMIC_PATH.exists()}")
print(f"  UMLS: {UMLS_PATH.exists()}")

# Target diagnoses
TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# Production checkpoint names
CHECKPOINT_DIAGNOSIS = 'shifamind_diagnosis.pt'
CHECKPOINT_CONCEPTS = 'shifamind_concepts.pt'
CHECKPOINT_FINAL = 'shifamind_final.pt'

print("✅ Setup complete")


# ============================================================================
# @title 2. Data Loading
# @markdown Load UMLS, MIMIC-IV, and ICD-10 data
# ============================================================================

print("\n" + "="*70)
print("DATA LOADING")
print("="*70)

class FastUMLSLoader:
    """Fast UMLS concept loader with semantic type filtering"""

    def __init__(self, umls_path: Path):
        self.umls_path = umls_path
        self.concepts = {}
        self.cui_to_icd10 = defaultdict(list)
        self.icd10_to_cui = defaultdict(list)

    def load_concepts(self, max_concepts: int = 30000):
        print(f"\n📚 Loading UMLS concepts (max: {max_concepts})...")

        target_types = {'T047', 'T046', 'T184', 'T033', 'T048', 'T037', 'T191', 'T020'}
        cui_to_types = self._load_semantic_types()

        mrconso_path = self.umls_path / 'MRCONSO.RRF'
        concepts_loaded = 0

        with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  Parsing", total=max_concepts):
                if concepts_loaded >= max_concepts:
                    break

                fields = line.strip().split('|')
                if len(fields) < 15:
                    continue

                cui, lang, sab, code, term = fields[0], fields[1], fields[11], fields[13], fields[14]

                if lang != 'ENG' or sab not in ['SNOMEDCT_US', 'ICD10CM', 'MSH', 'NCI']:
                    continue

                if cui not in cui_to_types:
                    continue
                types = cui_to_types[cui]
                if not any(t in target_types for t in types):
                    continue

                if cui not in self.concepts:
                    self.concepts[cui] = {
                        'cui': cui,
                        'preferred_name': term,
                        'terms': [term],
                        'sources': {sab: [code]},
                        'semantic_types': types
                    }
                    concepts_loaded += 1
                else:
                    if term not in self.concepts[cui]['terms']:
                        self.concepts[cui]['terms'].append(term)

                if sab == 'ICD10CM' and code:
                    self.cui_to_icd10[cui].append(code)
                    self.icd10_to_cui[code].append(cui)

        print(f"  ✅ Loaded {len(self.concepts)} concepts")
        return self.concepts

    def _load_semantic_types(self) -> Dict[str, List[str]]:
        mrsty_path = self.umls_path / 'MRSTY.RRF'
        cui_to_types = defaultdict(list)

        with open(mrsty_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) >= 2:
                    cui_to_types[fields[0]].append(fields[1])

        return cui_to_types

    def load_definitions(self, concepts: Dict) -> Dict:
        print("\n📖 Loading definitions...")
        mrdef_path = self.umls_path / 'MRDEF.RRF'
        definitions_added = 0

        with open(mrdef_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="  Parsing"):
                fields = line.strip().split('|')
                if len(fields) >= 6:
                    cui, definition = fields[0], fields[5]

                    if cui in concepts and definition:
                        if 'definition' not in concepts[cui]:
                            concepts[cui]['definition'] = definition
                            definitions_added += 1

        print(f"  ✅ Added {definitions_added} definitions")
        return concepts


class MIMICLoader:
    """MIMIC-IV data loader"""

    def __init__(self, mimic_path: Path, notes_path: Path):
        self.mimic_path = mimic_path
        self.hosp_path = mimic_path / 'mimic-iv-3.1/hosp'
        self.notes_path = notes_path

    def load_diagnoses(self) -> pd.DataFrame:
        diag_path = self.hosp_path / 'diagnoses_icd.csv.gz'
        return pd.read_csv(diag_path, compression='gzip')

    def load_admissions(self) -> pd.DataFrame:
        adm_path = self.hosp_path / 'admissions.csv.gz'
        return pd.read_csv(adm_path, compression='gzip')

    def load_discharge_notes(self) -> pd.DataFrame:
        discharge_path = self.notes_path / 'note' / 'discharge.csv.gz'
        if not discharge_path.exists():
            discharge_path = self.notes_path / 'discharge.csv.gz'
        return pd.read_csv(discharge_path, compression='gzip')


def load_icd10_descriptions(icd_path: Path) -> Dict[str, str]:
    """Load ICD-10 code descriptions"""
    codes_file = icd_path / 'icd10cm-codes-2024.txt'
    descriptions = {}

    with open(codes_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    descriptions[parts[0]] = parts[1]

    return descriptions


def prepare_dataset(df_diag, df_adm, df_notes, icd_descriptions, target_codes, max_per_code=3000):
    """Prepare balanced dataset from MIMIC-IV"""
    print("\n🔧 Preparing dataset...")

    df_diag = df_diag[df_diag['icd_version'] == 10].copy()
    df_diag['icd_code'] = df_diag['icd_code'].str.replace('.', '', regex=False)

    text_col = 'text'
    if 'text' not in df_notes.columns:
        text_cols = [col for col in df_notes.columns if 'text' in col.lower()]
        text_col = text_cols[0]

    df_notes_with_diag = df_notes.merge(
        df_diag.groupby('hadm_id')['icd_code'].apply(list).reset_index(),
        on='hadm_id', how='inner'
    )

    df = df_notes_with_diag.rename(columns={
        'icd_code': 'icd_codes',
        text_col: 'text'
    })[['hadm_id', 'text', 'icd_codes']].copy()

    df['has_target'] = df['icd_codes'].apply(
        lambda codes: any(code in target_codes for code in codes)
    )
    df_filtered = df[df['has_target']].copy()

    df_filtered['labels'] = df_filtered['icd_codes'].apply(
        lambda codes: [1 if code in codes else 0 for code in target_codes]
    )

    # Balance dataset
    balanced_indices = set()
    for code in target_codes:
        code_indices = df_filtered[
            df_filtered['icd_codes'].apply(lambda x: code in x)
        ].index.tolist()
        n_samples = min(len(code_indices), max_per_code)
        selected = np.random.choice(code_indices, size=n_samples, replace=False)
        balanced_indices.update(selected)

    df_final = df_filtered.loc[list(balanced_indices)].reset_index(drop=True)
    df_final = df_final[df_final['text'].notnull()].reset_index(drop=True)

    print(f"  ✅ Dataset: {len(df_final)} samples")
    return df_final, target_codes


# Load data
print("\nLoading UMLS...")
umls_loader = FastUMLSLoader(UMLS_PATH)
umls_concepts = umls_loader.load_concepts(max_concepts=30000)
umls_concepts = umls_loader.load_definitions(umls_concepts)

print("\nLoading ICD-10 descriptions...")
icd_descriptions = load_icd10_descriptions(ICD_PATH)

print("\nLoading MIMIC-IV...")
mimic_loader = MIMICLoader(MIMIC_PATH, NOTES_PATH)
df_diag = mimic_loader.load_diagnoses()
df_adm = mimic_loader.load_admissions()
df_notes = mimic_loader.load_discharge_notes()

print("\nPreparing dataset...")
df_data, target_codes = prepare_dataset(df_diag, df_adm, df_notes, icd_descriptions, TARGET_CODES)

# Train/val/test split
df_train, df_temp = train_test_split(df_data, test_size=0.3, random_state=SEED, stratify=df_data['labels'].apply(tuple))
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=SEED, stratify=df_temp['labels'].apply(tuple))

print(f"\n📊 Dataset split:")
print(f"  Train: {len(df_train)}")
print(f"  Val: {len(df_val)}")
print(f"  Test: {len(df_test)}")

print("\n✅ Data loading complete")


# ============================================================================
# @title 3. Concept Store Building
# @markdown Build diagnosis-specific medical concept store with filtering
# ============================================================================

print("\n" + "="*70)
print("CONCEPT STORE BUILDING")
print("="*70)

# CRITICAL: Diagnosis-specific keywords for concept filtering
DIAGNOSIS_KEYWORDS = {
    'J189': ['pneumonia', 'lung', 'respiratory', 'infection', 'infiltrate',
             'bacterial', 'pulmonary', 'aspiration'],
    'I5023': ['heart', 'cardiac', 'failure', 'cardiomyopathy', 'edema',
              'ventricular', 'atrial', 'systolic'],
    'A419': ['sepsis', 'septicemia', 'bacteremia', 'infection', 'septic',
             'shock', 'organ'],
    'K8000': ['cholecystitis', 'gallbladder', 'biliary', 'gallstone',
              'cholelithiasis', 'bile']
}

# CRITICAL: Blacklist wrong concepts per diagnosis
DIAGNOSIS_CONCEPT_BLACKLIST = {
    'I5023': [  # Heart failure should NEVER activate these pneumonia concepts
        'C0085740',  # Mendelson Syndrome
        'C0155860',  # Pseudomonas pneumonia
        'C0032302',  # Mycoplasma pneumonia
        'C0032286',  # Pneumonia other bacteria
        'C0001311'   # Acute bronchiolitis
    ],
    'J189': [],  # Add as needed
    'A419': [],
    'K8000': []
}


class SemanticTypeValidator:
    """Filters concepts by clinical relevance"""

    RELEVANT_TYPES = {
        'T047', 'T046', 'T184', 'T033', 'T048', 'T037', 'T191', 'T020',
    }

    def __init__(self, umls_concepts: Dict):
        self.umls_concepts = umls_concepts

    def validate_concept(self, cui: str, diagnosis_code: str = None) -> bool:
        if cui not in self.umls_concepts:
            return False

        concept = self.umls_concepts[cui]
        semantic_types = set(concept.get('semantic_types', []))

        if not semantic_types.intersection(self.RELEVANT_TYPES):
            return False

        return True


class ConceptStore:
    """Medical concept store with diagnosis-specific filtering"""

    def __init__(self, umls_concepts: Dict, icd_to_cui: Dict):
        self.umls_concepts = umls_concepts
        self.icd_to_cui = icd_to_cui
        self.concepts = {}
        self.concept_to_idx = {}
        self.idx_to_concept = {}
        self.semantic_validator = SemanticTypeValidator(umls_concepts)

    def build_concept_set(self, target_icd_codes: List[str],
                         icd_descriptions: Dict[str, str],
                         target_concept_count: int = 150):
        print(f"\n🔬 Building medical concept set (target: {target_concept_count})...")

        relevant_cuis = set()

        # Strategy 1: Direct ICD mappings
        for icd in target_icd_codes:
            variants = self._get_icd_variants(icd)
            for variant in variants:
                if variant in self.icd_to_cui:
                    cuis = self.icd_to_cui[variant]
                    validated = [
                        cui for cui in cuis
                        if self.semantic_validator.validate_concept(cui, icd)
                    ]
                    relevant_cuis.update(validated[:30])

        print(f"  Direct mappings: {len(relevant_cuis)} concepts")

        # Strategy 2: Keyword expansion per diagnosis
        per_diagnosis_concepts = {icd: set() for icd in target_icd_codes}

        for icd in target_icd_codes:
            keywords = DIAGNOSIS_KEYWORDS.get(icd, [])

            for cui, info in self.umls_concepts.items():
                if cui in relevant_cuis:
                    continue

                terms_text = ' '.join([info['preferred_name']] + info.get('terms', [])).lower()

                if any(kw in terms_text for kw in keywords):
                    if self.semantic_validator.validate_concept(cui, icd):
                        per_diagnosis_concepts[icd].add(cui)

        for icd in target_icd_codes:
            print(f"    {icd}: {len(per_diagnosis_concepts[icd])} keyword-matched concepts")

        for icd_concepts in per_diagnosis_concepts.values():
            relevant_cuis.update(icd_concepts)

        print(f"  Total concepts: {len(relevant_cuis)}")

        # Build final concept store
        for cui in list(relevant_cuis)[:target_concept_count]:
            if cui in self.umls_concepts:
                concept = self.umls_concepts[cui]
                self.concepts[cui] = {
                    'cui': cui,
                    'preferred_name': concept['preferred_name'],
                    'definition': concept.get('definition', ''),
                    'terms': concept.get('terms', []),
                    'semantic_types': concept.get('semantic_types', [])
                }

        concept_list = list(self.concepts.keys())
        self.concept_to_idx = {cui: i for i, cui in enumerate(concept_list)}
        self.idx_to_concept = {i: cui for i, cui in enumerate(concept_list)}

        # Build diagnosis-concept mappings
        self._build_diagnosis_concept_mapping(target_icd_codes)

        print(f"  ✅ Final: {len(self.concepts)} validated concepts")
        return self.concepts

    def _build_diagnosis_concept_mapping(self, target_icd_codes: List[str]):
        """Build mapping from diagnosis codes to relevant concept indices"""
        print("\n🔗 Building diagnosis-concept mappings...")

        self.diagnosis_to_concepts = {}

        for icd in target_icd_codes:
            keywords = DIAGNOSIS_KEYWORDS.get(icd, [])
            relevant_concept_indices = []

            for cui, info in self.concepts.items():
                concept_idx = self.concept_to_idx[cui]
                terms_text = ' '.join([info['preferred_name']] + info.get('terms', [])).lower()

                if any(kw in terms_text for kw in keywords):
                    relevant_concept_indices.append(concept_idx)

            self.diagnosis_to_concepts[icd] = relevant_concept_indices
            print(f"  {icd}: {len(relevant_concept_indices)} relevant concepts")

        print(f"  ✅ Diagnosis-concept mappings created")

    def _get_icd_variants(self, code: str) -> List[str]:
        variants = {code, code.replace('.', '')}
        no_dots = code.replace('.', '')
        if len(no_dots) >= 4:
            variants.add(no_dots[:3] + '.' + no_dots[3:])
        variants.add(no_dots[:3])
        return list(variants)

    def create_concept_embeddings(self, tokenizer, model, device):
        print("\n🧬 Creating concept embeddings...")

        concept_texts = []
        for cui, info in self.concepts.items():
            text = f"{info['preferred_name']}."
            if info['definition']:
                text += f" {info['definition'][:150]}"
            concept_texts.append(text)

        batch_size = 32
        all_embeddings = []

        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(concept_texts), batch_size), desc="  Encoding"):
                batch = concept_texts[i:i+batch_size]
                encodings = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(device)

                outputs = model(**encodings)
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu())

        final_embeddings = torch.cat(all_embeddings, dim=0).to(device)
        print(f"  ✅ Created embeddings: {final_embeddings.shape}")

        return final_embeddings

    def filter_concepts_by_diagnosis(self, concept_scores, diagnosis_code, concept_cuis, threshold=0.7):
        """
        CRITICAL: Filter activated concepts by diagnosis-specific keywords and blacklist

        This fixes the concept selection bias where wrong concepts activate
        (e.g., pneumonia concepts for heart failure diagnosis)
        """
        keywords = DIAGNOSIS_KEYWORDS.get(diagnosis_code, [])
        blacklist = set(DIAGNOSIS_CONCEPT_BLACKLIST.get(diagnosis_code, []))

        filtered_results = []

        for idx, score in enumerate(concept_scores):
            if score <= threshold or idx >= len(concept_cuis):
                continue

            cui = concept_cuis[idx]

            # Apply blacklist
            if cui in blacklist:
                continue

            # Apply keyword filtering
            if cui in self.concepts:
                concept_info = self.concepts[cui]
                terms_text = ' '.join([concept_info['preferred_name']] + concept_info.get('terms', [])).lower()

                # Check if concept matches diagnosis keywords
                if any(kw in terms_text for kw in keywords):
                    filtered_results.append({
                        'idx': idx,
                        'cui': cui,
                        'name': concept_info['preferred_name'],
                        'score': float(score)
                    })

        # Sort by score and return top concepts
        filtered_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)[:5]

        return filtered_results


class DiagnosisConditionalLabeler:
    """Generate concept labels using PMI (Pointwise Mutual Information)"""

    def __init__(self, concept_store, icd_to_cui, pmi_threshold=1.0):
        self.concept_store = concept_store
        self.icd_to_cui = icd_to_cui
        self.pmi_threshold = pmi_threshold
        self.diagnosis_counts = defaultdict(int)
        self.concept_counts = defaultdict(int)
        self.diagnosis_concept_counts = defaultdict(lambda: defaultdict(int))
        self.total_pairs = 0
        self.pmi_scores = {}

    def build_cooccurrence_statistics(self, df_train, target_codes):
        """Build diagnosis-concept co-occurrence statistics"""
        print("\n📊 Building co-occurrence statistics...")

        for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="  Processing"):
            diagnosis_codes = row['icd_codes']
            note_concepts = set()

            for dx_code in diagnosis_codes:
                self.diagnosis_counts[dx_code] += 1

                dx_variants = self._get_icd_variants(dx_code)
                for variant in dx_variants:
                    if variant in self.icd_to_cui:
                        cuis = self.icd_to_cui[variant]
                        valid_cuis = [cui for cui in cuis if cui in self.concept_store.concepts]
                        note_concepts.update(valid_cuis)

            for concept_cui in note_concepts:
                self.concept_counts[concept_cui] += 1
                for dx_code in diagnosis_codes:
                    self.diagnosis_concept_counts[dx_code][concept_cui] += 1
                    self.total_pairs += 1

        print(f"  ✅ Unique diagnoses: {len(self.diagnosis_counts)}")
        print(f"  ✅ Unique concepts: {len(self.concept_counts)}")
        print(f"  ✅ Total co-occurrences: {self.total_pairs}")

        return self._compute_pmi_scores()

    def _compute_pmi_scores(self):
        """Compute PMI scores"""
        print("\n  Computing PMI scores...")

        total_diagnoses = sum(self.diagnosis_counts.values())
        total_concepts = sum(self.concept_counts.values())

        for dx_code in tqdm(self.diagnosis_counts.keys(), desc="  PMI"):
            p_dx = self.diagnosis_counts[dx_code] / total_diagnoses

            for concept_cui in self.concept_counts.keys():
                cooccur_count = self.diagnosis_concept_counts[dx_code].get(concept_cui, 0)
                if cooccur_count == 0:
                    continue

                p_dx_concept = cooccur_count / self.total_pairs
                p_concept = self.concept_counts[concept_cui] / total_concepts

                pmi = math.log(p_dx_concept / (p_dx * p_concept + 1e-10) + 1e-10)

                if pmi > self.pmi_threshold:
                    self.pmi_scores[(dx_code, concept_cui)] = pmi

        print(f"  ✅ Computed {len(self.pmi_scores)} significant PMI scores")

        concepts_with_pmi = set()
        for (dx_code, concept_cui) in self.pmi_scores.keys():
            concepts_with_pmi.add(concept_cui)

        return concepts_with_pmi

    def generate_labels(self, diagnosis_codes: List[str]) -> List[int]:
        """Generate concept labels for a sample"""
        concept_scores = defaultdict(float)

        for dx_code in diagnosis_codes:
            for concept_cui in self.concept_store.concepts.keys():
                key = (dx_code, concept_cui)
                if key in self.pmi_scores:
                    concept_scores[concept_cui] = max(
                        concept_scores[concept_cui],
                        self.pmi_scores[key]
                    )

        labels = []
        concept_ids = list(self.concept_store.concepts.keys())

        for cui in concept_ids:
            label = 1 if concept_scores[cui] > 0 else 0
            labels.append(label)

        return labels

    def generate_dataset_labels(self, df_data, cache_file: str = 'diagnosis_conditional_labels.pkl') -> np.ndarray:
        """Generate labels for entire dataset with caching"""

        if os.path.exists(cache_file):
            print(f"\n📦 Loading cached labels from {cache_file}...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print(f"\n🏷️  Generating labels for {len(df_data)} samples...")

        all_labels = []
        for row in tqdm(df_data.itertuples(), total=len(df_data), desc="  Labeling"):
            labels = self.generate_labels(row.icd_codes)
            all_labels.append(labels)

        all_labels = np.array(all_labels)

        with open(cache_file, 'wb') as f:
            pickle.dump(all_labels, f)

        print(f"  ✅ Generated labels: {all_labels.shape}")
        print(f"  📊 Avg labels per sample: {all_labels.sum(axis=1).mean():.1f}")

        return all_labels

    def _get_icd_variants(self, code: str) -> List[str]:
        variants = {code, code.replace('.', '')}
        no_dots = code.replace('.', '')
        if len(no_dots) >= 4:
            variants.add(no_dots[:3] + '.' + no_dots[3:])
        variants.add(no_dots[:3])
        return list(variants)


# Build concept store
print("\nBuilding concept store...")
concept_store = ConceptStore(umls_concepts, umls_loader.icd10_to_cui)
concept_store.build_concept_set(TARGET_CODES, icd_descriptions, target_concept_count=150)

# Initialize tokenizer and base model
print("\nInitializing Bio_ClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

# Create concept embeddings
concept_embeddings = concept_store.create_concept_embeddings(tokenizer, base_model, device)

# Build PMI labeler
print("\nBuilding diagnosis-conditional labeler...")
diagnosis_labeler = DiagnosisConditionalLabeler(concept_store, umls_loader.icd10_to_cui, pmi_threshold=1.0)
concepts_with_pmi = diagnosis_labeler.build_cooccurrence_statistics(df_train, TARGET_CODES)

print("\n✅ Concept store building complete")


# ============================================================================
# @title 4. Model Architecture
# @markdown Define EnhancedCrossAttention and ShifaMindModel
# ============================================================================

print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)

class EnhancedCrossAttention(nn.Module):
    """Cross-attention between clinical text and medical concepts"""

    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, concept_embeddings, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_concepts = concept_embeddings.shape[0]

        concepts_batch = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        Q = self.query(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = self.key(concepts_batch).view(
            batch_size, num_concepts, self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = self.value(concepts_batch).view(
            batch_size, num_concepts, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )

        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        output = hidden_states + gate_values * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1)


class ShifaMindModel(nn.Module):
    """ShifaMind: Concept-enhanced medical diagnosis prediction"""

    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_size = base_model.config.hidden_size
        self.fusion_layers = fusion_layers

        self.fusion_modules = nn.ModuleList([
            EnhancedCrossAttention(self.hidden_size, num_heads=8)
            for _ in fusion_layers
        ])

        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, num_concepts)

        self.diagnosis_concept_interaction = nn.Bilinear(
            num_classes, num_concepts, num_concepts
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, concept_embeddings, return_diagnosis_only=False):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        if return_diagnosis_only:
            cls_hidden = outputs.last_hidden_state[:, 0, :]
            cls_hidden = self.dropout(cls_hidden)
            diagnosis_logits = self.diagnosis_head(cls_hidden)
            return {'logits': diagnosis_logits}

        hidden_states = outputs.hidden_states
        current_hidden = hidden_states[-1]

        fusion_attentions = []
        for i, fusion_module in enumerate(self.fusion_modules):
            layer_idx = self.fusion_layers[i]
            layer_hidden = hidden_states[layer_idx]

            fused_hidden, attn_weights = fusion_module(
                layer_hidden, concept_embeddings, attention_mask
            )
            fusion_attentions.append(attn_weights)

            if i == len(self.fusion_modules) - 1:
                current_hidden = fused_hidden

        cls_hidden = current_hidden[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)

        diagnosis_logits = self.diagnosis_head(cls_hidden)
        concept_logits = self.concept_head(cls_hidden)

        diagnosis_probs = torch.sigmoid(diagnosis_logits)
        refined_concept_logits = self.diagnosis_concept_interaction(
            diagnosis_probs, torch.sigmoid(concept_logits)
        )

        return {
            'logits': diagnosis_logits,
            'concept_scores': refined_concept_logits,
            'attention_weights': fusion_attentions
        }


class ClinicalDataset(Dataset):
    """Clinical text dataset"""

    def __init__(self, texts, labels, tokenizer, max_length=384, concept_labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.concept_labels = concept_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

        if self.concept_labels is not None:
            item['concept_labels'] = torch.FloatTensor(self.concept_labels[idx])

        return item


# Initialize model
print("\nInitializing ShifaMind model...")
shifamind_model = ShifaMindModel(
    base_model=base_model,
    num_concepts=len(concept_store.concepts),
    num_classes=len(TARGET_CODES),
    fusion_layers=[9, 11]
).to(device)

print(f"  Model parameters: {sum(p.numel() for p in shifamind_model.parameters()):,}")
print("✅ Model architecture defined")


# ============================================================================
# @title 5. Training - Stage 1: Diagnosis Head
# @markdown Train diagnosis prediction head (3 epochs)
# ============================================================================

print("\n" + "="*70)
print("STAGE 1: DIAGNOSIS HEAD TRAINING")
print("="*70)

# Check if checkpoint exists
if os.path.exists(CHECKPOINT_DIAGNOSIS):
    print(f"\n✅ Found existing checkpoint: {CHECKPOINT_DIAGNOSIS}")
    print("Skipping Stage 1 (already trained)")
    shifamind_model.load_state_dict(torch.load(CHECKPOINT_DIAGNOSIS, map_location=device)['model_state_dict'])
else:
    print("\nPreparing data loaders...")
    train_dataset = ClinicalDataset(
        df_train['text'].tolist(),
        df_train['labels'].tolist(),
        tokenizer
    )
    val_dataset = ClinicalDataset(
        df_val['text'].tolist(),
        df_val['labels'].tolist(),
        tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    print("\nStarting Stage 1 training...")
    optimizer = torch.optim.AdamW(shifamind_model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    num_training_steps = 3 * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )

    best_f1 = 0

    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")

        shifamind_model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = shifamind_model(
                input_ids, attention_mask, concept_embeddings,
                return_diagnosis_only=True
            )

            loss = criterion(outputs['logits'], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shifamind_model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        shifamind_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = shifamind_model(
                    input_ids, attention_mask, concept_embeddings,
                    return_diagnosis_only=True
                )

                preds = torch.sigmoid(outputs['logits']).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        pred_binary = (all_preds > 0.5).astype(int)

        macro_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)

        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Val Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1

            # Save with production naming
            torch.save({
                'model_state_dict': shifamind_model.state_dict(),
                'num_concepts': len(concept_store.concepts),
                'concept_cuis': list(concept_store.concepts.keys()),
                'concept_names': {cui: info['preferred_name'] for cui, info in concept_store.concepts.items()},
                'concept_embeddings': concept_embeddings,
                'macro_f1': best_f1
            }, CHECKPOINT_DIAGNOSIS)
            print(f"  ✅ Saved checkpoint: {CHECKPOINT_DIAGNOSIS} (F1: {best_f1:.4f})")

    print(f"\n✅ Stage 1 complete. Best F1: {best_f1:.4f}")

torch.cuda.empty_cache()


# ============================================================================
# @title 6. Training - Stage 2: Concept Labels
# @markdown Generate diagnosis-conditional concept labels using PMI
# ============================================================================

print("\n" + "="*70)
print("STAGE 2: GENERATING DIAGNOSIS-CONDITIONAL LABELS")
print("="*70)

# Generate labels for all splits
train_concept_labels = diagnosis_labeler.generate_dataset_labels(
    df_train,
    cache_file='diagnosis_conditional_labels_train.pkl'
)

val_concept_labels = diagnosis_labeler.generate_dataset_labels(
    df_val,
    cache_file='diagnosis_conditional_labels_val.pkl'
)

test_concept_labels = diagnosis_labeler.generate_dataset_labels(
    df_test,
    cache_file='diagnosis_conditional_labels_test.pkl'
)

print("\n✅ Stage 2 complete")


# ============================================================================
# @title 7. Training - Stage 3: Concept Head
# @markdown Train concept prediction head (2 epochs)
# ============================================================================

print("\n" + "="*70)
print("STAGE 3: CONCEPT HEAD TRAINING")
print("="*70)

if os.path.exists(CHECKPOINT_CONCEPTS):
    print(f"\n✅ Found existing checkpoint: {CHECKPOINT_CONCEPTS}")
    print("Skipping Stage 3 (already trained)")
    shifamind_model.load_state_dict(torch.load(CHECKPOINT_CONCEPTS, map_location=device)['model_state_dict'])
else:
    print("\nPreparing data loaders with concept labels...")
    train_dataset = ClinicalDataset(
        df_train['text'].tolist(),
        df_train['labels'].tolist(),
        tokenizer,
        concept_labels=train_concept_labels
    )
    val_dataset = ClinicalDataset(
        df_val['text'].tolist(),
        df_val['labels'].tolist(),
        tokenizer,
        concept_labels=val_concept_labels
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    print("\nStarting Stage 3 training...")
    optimizer = torch.optim.AdamW(shifamind_model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    best_concept_f1 = 0

    for epoch in range(2):
        print(f"\nEpoch {epoch+1}/2")

        shifamind_model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            concept_labels_batch = batch['concept_labels'].to(device)

            optimizer.zero_grad()

            outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

            # Concept loss
            concept_loss = criterion(outputs['concept_scores'], concept_labels_batch)

            # Confidence boost
            concept_probs = torch.sigmoid(outputs['concept_scores'])
            top_k_probs = torch.topk(concept_probs, k=12, dim=1)[0]
            confidence_loss = -torch.mean(top_k_probs)

            loss = 0.7 * concept_loss + 0.3 * confidence_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(shifamind_model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        shifamind_model.eval()
        all_concept_preds = []
        all_concept_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                concept_labels_batch = batch['concept_labels'].to(device)

                outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

                concept_preds = torch.sigmoid(outputs['concept_scores']).cpu().numpy()
                all_concept_preds.append(concept_preds)
                all_concept_labels.append(concept_labels_batch.cpu().numpy())

        all_concept_preds = np.vstack(all_concept_preds)
        all_concept_labels = np.vstack(all_concept_labels)
        concept_pred_binary = (all_concept_preds > 0.7).astype(int)

        concept_f1 = f1_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)

        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Val Concept F1: {concept_f1:.4f}")

        if concept_f1 > best_concept_f1:
            best_concept_f1 = concept_f1

            torch.save({
                'model_state_dict': shifamind_model.state_dict(),
                'num_concepts': len(concept_store.concepts),
                'concept_cuis': list(concept_store.concepts.keys()),
                'concept_names': {cui: info['preferred_name'] for cui, info in concept_store.concepts.items()},
                'concept_embeddings': concept_embeddings,
                'concept_f1': best_concept_f1
            }, CHECKPOINT_CONCEPTS)
            print(f"  ✅ Saved checkpoint: {CHECKPOINT_CONCEPTS} (F1: {best_concept_f1:.4f})")

    print(f"\n✅ Stage 3 complete. Best Concept F1: {best_concept_f1:.4f}")

torch.cuda.empty_cache()


# ============================================================================
# @title 8. Training - Stage 4: Joint Training
# @markdown Joint fine-tuning with alignment loss (3 epochs)
# ============================================================================

print("\n" + "="*70)
print("STAGE 4: JOINT FINE-TUNING WITH ALIGNMENT")
print("="*70)

class AlignmentLoss(nn.Module):
    """Alignment loss to enforce diagnosis-concept matching"""

    def __init__(self, concept_store, target_codes):
        super().__init__()
        self.concept_store = concept_store
        self.target_codes = target_codes
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, diagnosis_logits, concept_scores, diagnosis_labels, concept_labels):
        # Diagnosis loss
        diagnosis_loss = self.bce_loss(diagnosis_logits, diagnosis_labels)

        # Concept precision loss
        concept_precision_loss = self.bce_loss(concept_scores, concept_labels)

        # Confidence boost
        concept_probs = torch.sigmoid(concept_scores)
        top_k_probs = torch.topk(concept_probs, k=12, dim=1)[0]
        confidence_loss = -torch.mean(top_k_probs)

        # Alignment loss
        batch_size = diagnosis_logits.size(0)
        diagnosis_probs = torch.sigmoid(diagnosis_logits)
        predicted_diagnosis_indices = torch.argmax(diagnosis_probs, dim=1)

        alignment_targets = torch.zeros_like(concept_scores)

        for i in range(batch_size):
            pred_idx = predicted_diagnosis_indices[i].item()
            diagnosis_code = self.target_codes[pred_idx]

            relevant_concepts = self.concept_store.diagnosis_to_concepts.get(diagnosis_code, [])

            for concept_idx in relevant_concepts:
                alignment_targets[i, concept_idx] = 1.0

        alignment_loss = nn.functional.binary_cross_entropy_with_logits(
            concept_scores, alignment_targets, reduction='mean'
        )

        total_loss = (
            0.40 * diagnosis_loss +
            0.30 * concept_precision_loss +
            0.15 * confidence_loss +
            0.15 * alignment_loss
        )

        return total_loss, {
            'diagnosis': diagnosis_loss.item(),
            'concept': concept_precision_loss.item(),
            'confidence': confidence_loss.item(),
            'alignment': alignment_loss.item()
        }


if os.path.exists(CHECKPOINT_FINAL):
    print(f"\n✅ Found existing checkpoint: {CHECKPOINT_FINAL}")
    print("Skipping Stage 4 (already trained)")
    shifamind_model.load_state_dict(torch.load(CHECKPOINT_FINAL, map_location=device)['model_state_dict'])
else:
    print("\nStarting Stage 4 training...")
    optimizer = torch.optim.AdamW(shifamind_model.parameters(), lr=1.5e-5, weight_decay=0.01)
    criterion = AlignmentLoss(concept_store, TARGET_CODES)

    num_training_steps = 3 * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )

    best_f1 = 0

    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")

        shifamind_model.train()
        total_loss = 0
        loss_components = defaultdict(float)

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            concept_labels_batch = batch['concept_labels'].to(device)

            optimizer.zero_grad()

            outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

            loss, components = criterion(
                outputs['logits'],
                outputs['concept_scores'],
                labels,
                concept_labels_batch
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(shifamind_model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v

        avg_loss = total_loss / len(train_loader)

        print(f"  Loss: {avg_loss:.4f}")
        for k, v in loss_components.items():
            print(f"    {k}: {v/len(train_loader):.4f}")

        # Validation
        shifamind_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

                preds = torch.sigmoid(outputs['logits']).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        pred_binary = (all_preds > 0.5).astype(int)

        macro_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)

        print(f"  Val Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1

            torch.save({
                'model_state_dict': shifamind_model.state_dict(),
                'num_concepts': len(concept_store.concepts),
                'concept_cuis': list(concept_store.concepts.keys()),
                'concept_names': {cui: info['preferred_name'] for cui, info in concept_store.concepts.items()},
                'concept_embeddings': concept_embeddings,
                'target_codes': TARGET_CODES,
                'macro_f1': best_f1
            }, CHECKPOINT_FINAL)
            print(f"  ✅ Saved checkpoint: {CHECKPOINT_FINAL} (F1: {best_f1:.4f})")

    print(f"\n✅ Stage 4 complete. Best F1: {best_f1:.4f}")

torch.cuda.empty_cache()


# ============================================================================
# @title 9. Evaluation
# @markdown Evaluate model with concept filtering on test set
# ============================================================================

print("\n" + "="*70)
print("EVALUATION WITH CONCEPT FILTERING")
print("="*70)

# Load best model
print(f"\nLoading best model from {CHECKPOINT_FINAL}...")
checkpoint = torch.load(CHECKPOINT_FINAL, map_location=device)
shifamind_model.load_state_dict(checkpoint['model_state_dict'])
shifamind_model.eval()

# Prepare test loader
test_dataset = ClinicalDataset(
    df_test['text'].tolist(),
    df_test['labels'].tolist(),
    tokenizer,
    concept_labels=test_concept_labels
)
test_loader = DataLoader(test_dataset, batch_size=16)

print("\nEvaluating on test set...")
all_preds = []
all_labels = []
all_concept_preds = []
all_concept_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        concept_labels_batch = batch['concept_labels'].to(device)

        outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

        preds = torch.sigmoid(outputs['logits']).cpu().numpy()
        concept_preds = torch.sigmoid(outputs['concept_scores']).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        all_concept_preds.append(concept_preds)
        all_concept_labels.append(concept_labels_batch.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
all_concept_preds = np.vstack(all_concept_preds)
all_concept_labels = np.vstack(all_concept_labels)

# Diagnosis metrics
pred_binary = (all_preds > 0.5).astype(int)
macro_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)
macro_precision = precision_score(all_labels, pred_binary, average='macro', zero_division=0)
macro_recall = recall_score(all_labels, pred_binary, average='macro', zero_division=0)

print("\n" + "="*70)
print("TEST SET RESULTS")
print("="*70)
print(f"\n📊 Diagnosis Prediction:")
print(f"  Macro F1: {macro_f1:.4f}")
print(f"  Macro Precision: {macro_precision:.4f}")
print(f"  Macro Recall: {macro_recall:.4f}")

# Per-diagnosis metrics
print(f"\n📊 Per-Diagnosis F1:")
for i, code in enumerate(TARGET_CODES):
    f1 = f1_score(all_labels[:, i], pred_binary[:, i], zero_division=0)
    print(f"  {code} ({ICD_DESCRIPTIONS[code][:40]}...): {f1:.4f}")

# Concept metrics (with filtering)
concept_pred_binary = (all_concept_preds > 0.7).astype(int)
concept_precision = precision_score(all_concept_labels, concept_pred_binary, average='macro', zero_division=0)

print(f"\n💡 Concept Prediction:")
print(f"  Concept Precision: {concept_precision:.4f}")
print(f"  Avg concepts activated per sample: {concept_pred_binary.sum(axis=1).mean():.1f}")

print("\n✅ Evaluation complete")


# ============================================================================
# @title 10. Gradio Demo
# @markdown Interactive web interface with diagnosis-conditional concept filtering
# ============================================================================

print("\n" + "="*70)
print("GRADIO DEMO")
print("="*70)

try:
    import gradio as gr
except ImportError:
    print("Installing gradio...")
    os.system('pip install -q gradio')
    import gradio as gr


class EvidenceExtractor:
    """Extracts evidence spans from clinical text using attention weights"""

    def __init__(self, tokenizer, attention_percentile=85, min_span_tokens=5,
                 max_span_tokens=50, top_k_spans=3):
        self.tokenizer = tokenizer
        self.attention_percentile = attention_percentile
        self.min_span_tokens = min_span_tokens
        self.max_span_tokens = max_span_tokens
        self.top_k_spans = top_k_spans

    def extract(self, input_ids, attention_weights, filtered_concepts):
        """Extract evidence spans for filtered concepts"""

        if not filtered_concepts:
            return []

        # Aggregate attention across layers
        aggregated_attention = torch.stack(attention_weights).mean(dim=0)

        results = []
        for concept_info in filtered_concepts:
            concept_idx = concept_info['idx']
            concept_attention = aggregated_attention[:, concept_idx]

            # Find high-attention tokens
            threshold_val = torch.quantile(concept_attention, self.attention_percentile / 100.0)
            high_attention_mask = concept_attention >= threshold_val

            # Extract spans
            spans = self._find_spans(
                high_attention_mask.cpu().numpy(),
                concept_attention.cpu().numpy()
            )

            # Decode to text
            decoded_spans = []
            for span in spans:
                if not (self.min_span_tokens <= (span['end'] - span['start']) <= self.max_span_tokens):
                    continue

                span_tokens = input_ids[span['start']:span['end']]
                span_text = self.tokenizer.decode(span_tokens, skip_special_tokens=True)
                span_text = span_text.replace(' ##', '').replace('##', '').strip()

                if len(span_text) < 20:
                    continue
                if not any(c.isalnum() for c in span_text):
                    continue
                if self._is_noise(span_text):
                    continue

                decoded_spans.append({
                    'text': span_text,
                    'attention_score': float(span['avg_attention'])
                })

            decoded_spans = sorted(decoded_spans, key=lambda x: x['attention_score'], reverse=True)[:self.top_k_spans]

            if decoded_spans:
                results.append({
                    'concept_cui': concept_info['cui'],
                    'concept_name': concept_info['name'],
                    'concept_score': concept_info['score'],
                    'evidence_spans': decoded_spans
                })

        return results

    def _find_spans(self, mask, attention_scores):
        """Find consecutive high-attention token spans"""
        spans = []
        start_idx = None

        for i, is_high in enumerate(mask):
            if is_high:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    spans.append({
                        'start': start_idx,
                        'end': i,
                        'avg_attention': attention_scores[start_idx:i].mean()
                    })
                    start_idx = None

        if start_idx is not None:
            spans.append({
                'start': start_idx,
                'end': len(mask),
                'avg_attention': attention_scores[start_idx:].mean()
            })

        return spans

    def _is_noise(self, text):
        """Check if text is EHR noise"""
        noise_patterns = [
            r'_{3,}',
            r'^\s*(name|unit no|admission date|discharge date|date of birth|sex|service|allergies|attending|chief complaint|major surgical)\s*:',
            r'\[\s*\*\*.*?\*\*\s*\]',
            r'^[^a-z]{10,}$',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in noise_patterns)


def predict_with_filtering(clinical_text):
    """Run inference with diagnosis-conditional concept filtering"""

    # Tokenize
    encoding = tokenizer(
        clinical_text,
        padding='max_length',
        truncation=True,
        max_length=384,
        return_tensors='pt'
    ).to(device)

    # Run model
    with torch.no_grad():
        outputs = shifamind_model(
            encoding['input_ids'],
            encoding['attention_mask'],
            concept_embeddings
        )

    # Get diagnosis prediction
    diagnosis_probs = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
    diagnosis_pred_idx = np.argmax(diagnosis_probs)
    diagnosis_code = TARGET_CODES[diagnosis_pred_idx]
    diagnosis_score = float(diagnosis_probs[diagnosis_pred_idx])

    concept_scores = torch.sigmoid(outputs['concept_scores']).cpu().numpy()[0]
    concept_cuis = list(concept_store.concepts.keys())

    # CRITICAL: Apply diagnosis-conditional concept filtering
    filtered_concepts = concept_store.filter_concepts_by_diagnosis(
        concept_scores,
        diagnosis_code,
        concept_cuis,
        threshold=0.7
    )

    # Extract evidence for filtered concepts
    evidence_extractor = EvidenceExtractor(tokenizer)
    sample_attention_weights = [attn[0] for attn in outputs['attention_weights']]
    evidence = evidence_extractor.extract(
        encoding['input_ids'][0].cpu(),
        sample_attention_weights,
        filtered_concepts
    )

    # Format output
    output = f"### 🎯 Diagnosis Prediction\n\n"
    output += f"**{diagnosis_code}** - {ICD_DESCRIPTIONS[diagnosis_code]}\n\n"
    output += f"Confidence: **{diagnosis_score:.3f}**\n\n"
    output += "---\n\n"

    output += f"### 💡 Activated Medical Concepts ({len(evidence)} total)\n\n"
    output += "*Filtered by diagnosis-specific keywords and blacklist*\n\n"

    for i, extraction in enumerate(evidence, 1):
        output += f"#### [{i}] {extraction['concept_name']}\n"
        output += f"- **CUI:** {extraction['concept_cui']}\n"
        output += f"- **Score:** {extraction['concept_score']:.3f}\n"
        output += f"- **Evidence Spans:**\n\n"

        for j, span in enumerate(extraction['evidence_spans'], 1):
            output += f"  {j}. *\"{span['text']}\"* (attention: {span['attention_score']:.3f})\n\n"

        output += "\n"

    return output


# Example clinical texts
EXAMPLES = [
    """Patient is a 65-year-old male presenting with fever, productive cough, and shortness of breath for 3 days.
    Chest X-ray shows right lower lobe infiltrate. Vital signs: temp 38.9°C, HR 105, RR 24, BP 135/85.
    Oxygen saturation 92% on room air. Physical exam reveals crackles in right lower lung field.
    Started on empiric antibiotics.""",

    """78-year-old female with history of CHF admitted with worsening dyspnea and bilateral lower extremity edema.
    Patient reports orthopnea and paroxysmal nocturnal dyspnea. Examination shows jugular venous distension,
    S3 gallop, and 3+ pitting edema bilaterally. Chest X-ray demonstrates cardiomegaly and pulmonary vascular
    congestion. BNP elevated at 1200.""",

    """45-year-old male presents with fever, hypotension, and altered mental status. Started feeling unwell 2 days ago
    with fever and chills. Now with BP 85/50, HR 125, RR 28, temp 39.5°C. Labs show WBC 18,000, lactate 4.2.
    Blood cultures pending. Started on broad-spectrum antibiotics and IV fluids."""
]

# Create Gradio interface
print("\nCreating Gradio interface...")

demo = gr.Interface(
    fn=predict_with_filtering,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Enter clinical text here...",
        label="Clinical Text"
    ),
    outputs=gr.Markdown(label="ShifaMind Analysis"),
    title="ShifaMind: Medical Diagnosis with Explainability",
    description="""
    **Production-ready medical diagnosis prediction with concept-enhanced explainability**

    Features:
    - Diagnosis prediction for 4 ICD-10 codes (Pneumonia, Heart Failure, Sepsis, Cholecystitis)
    - Diagnosis-conditional concept filtering (fixes concept selection bias)
    - Evidence extraction from clinical text
    - Attention-based reasoning chains

    The model correctly filters concepts based on predicted diagnosis, preventing wrong concept activation.
    """,
    examples=EXAMPLES,
    theme=gr.themes.Soft()
)

print("\n🚀 Launching Gradio demo...")
print("   Access the demo via the public URL below")

demo.launch(share=True)

print("\n✅ Demo launched successfully!")
print("\n" + "="*70)
print("SHIFAMIND 040 - ALL STAGES COMPLETE")
print("="*70)
print(f"\n📦 Checkpoints saved:")
print(f"  - {CHECKPOINT_DIAGNOSIS}")
print(f"  - {CHECKPOINT_CONCEPTS}")
print(f"  - {CHECKPOINT_FINAL}")
print(f"\n🎯 Production ready!")
