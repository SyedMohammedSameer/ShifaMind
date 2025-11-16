#!/usr/bin/env python3
"""
ShifaMind Phase 1: Focal Loss & Training Improvements
Building on 016.py (F1: 0.7759 → Target: 0.81+)

BASELINE FROM 016.py:
✅ Bio_ClinicalBERT with Phase 4 Revised architecture
✅ F1 = 0.7759
✅ 150 concepts with PMI-based diagnosis-conditional labeling
✅ Multi-layer cross-attention fusion (layers 9, 11)
✅ 4-stage training approach

PHASE 1 IMPROVEMENTS:
1. ✅ Focal Loss (alpha=0.25, gamma=2.0) replaces BCEWithLogitsLoss
   - Better handling of class imbalance
   - Focuses on hard-to-classify examples

2. ✅ More Training: 8 → 15 epochs with early stopping
   - Stage 1: 3 → 5 epochs
   - Stage 3: 2 → 3 epochs
   - Stage 4: 3 → 7 epochs
   - Early stopping patience=3

3. ✅ Cosine Annealing with Warmup
   - Replaces linear schedule
   - Better convergence

4. ✅ More Concepts: 150 → 300
   - Manageable increase
   - Better coverage

Expected Results:
- F1: 0.78-0.82
- Concepts: 15-20 per sample
- Precision: 70-80%
"""

# ============================================================================
# SETUP & IMPORTS
# ============================================================================
import os
import sys
import warnings
warnings.filterwarnings('ignore')

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    print("🔧 Installing dependencies...")
    os.system('pip install -q faiss-cpu scikit-learn transformers torch')
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
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
import time
import pickle
import math

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

# ============================================================================
# DATA PATHS
# ============================================================================
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind/01_Raw_Datasets/Extracted')
MIMIC_PATH = BASE_PATH / 'mimic-iv-3.1'
UMLS_PATH = BASE_PATH / 'umls-2025AA-metathesaurus-full/2025AA/META'
ICD_PATH = BASE_PATH / 'icd10cm-CodesDescriptions-2024'
NOTES_PATH = BASE_PATH / 'mimic-iv-note-2.2'

print(f"\n📁 Data paths:")
print(f"  MIMIC-IV: {MIMIC_PATH.exists()}")
print(f"  UMLS: {UMLS_PATH.exists()}")

# ============================================================================
# PHASE 1: FOCAL LOSS IMPLEMENTATION
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor in [0,1] to balance positive/negative examples
        gamma: Focusing parameter for modulating loss (gamma >= 0)

    References:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (before sigmoid)
            targets: Ground truth labels (0 or 1)
        """
        # Get probabilities
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# PHASE 4 REVISED: DIAGNOSIS-CONDITIONAL CONCEPT LABELER
# ============================================================================
class DiagnosisConditionalLabeler:
    """
    Generate concept labels based on diagnosis-concept co-occurrence

    Uses Pointwise Mutual Information (PMI) to find concepts that
    frequently co-occur with specific diagnoses in training data.

    Much more reliable than semantic similarity for clinical notes.
    """

    def __init__(self, concept_store, icd_to_cui, pmi_threshold=1.0):
        self.concept_store = concept_store
        self.icd_to_cui = icd_to_cui
        self.pmi_threshold = pmi_threshold

        # Co-occurrence statistics
        self.diagnosis_counts = defaultdict(int)
        self.concept_counts = defaultdict(int)
        self.diagnosis_concept_counts = defaultdict(lambda: defaultdict(int))
        self.total_pairs = 0

        # PMI scores
        self.pmi_scores = {}

        print(f"\n🏷️  Initializing Diagnosis-Conditional Labeler...")
        print(f"  PMI Threshold: {pmi_threshold}")
        print(f"  Concepts: {len(concept_store.concepts)}")

    def build_cooccurrence_statistics(self, df_train, target_codes):
        """
        Build diagnosis-concept co-occurrence from training data

        For each clinical note:
        - Extract diagnosis codes
        - Map diagnoses to UMLS concepts via ICD mappings
        - Count co-occurrences
        """
        print("\n📊 Building diagnosis-concept co-occurrence statistics...")

        # Get all ICD codes from training data
        all_icd_codes = []
        for codes in df_train['icd_codes']:
            all_icd_codes.extend(codes)

        print(f"  Total diagnosis instances: {len(all_icd_codes)}")

        # Build co-occurrence matrix
        samples_processed = 0

        for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="  Processing"):
            diagnosis_codes = row['icd_codes']

            # Get concepts for each diagnosis via ICD mapping
            note_concepts = set()

            for dx_code in diagnosis_codes:
                # Count diagnosis
                self.diagnosis_counts[dx_code] += 1

                # Get concepts mapped to this diagnosis
                dx_variants = self._get_icd_variants(dx_code)
                for variant in dx_variants:
                    if variant in self.icd_to_cui:
                        cuis = self.icd_to_cui[variant]
                        # Only use concepts in our concept store
                        valid_cuis = [cui for cui in cuis if cui in self.concept_store.concepts]
                        note_concepts.update(valid_cuis)

            # Count concept occurrences and co-occurrences
            for concept_cui in note_concepts:
                self.concept_counts[concept_cui] += 1

                # Co-occurrence with each diagnosis in this note
                for dx_code in diagnosis_codes:
                    self.diagnosis_concept_counts[dx_code][concept_cui] += 1
                    self.total_pairs += 1

            samples_processed += 1

        print(f"  ✅ Processed {samples_processed} samples")
        print(f"  Unique diagnoses: {len(self.diagnosis_counts)}")
        print(f"  Unique concepts: {len(self.concept_counts)}")
        print(f"  Total co-occurrences: {self.total_pairs}")

        # Compute PMI scores
        self._compute_pmi_scores()

    def _compute_pmi_scores(self):
        """
        Compute Pointwise Mutual Information (PMI) scores

        PMI(diagnosis, concept) = log( P(d,c) / (P(d) * P(c)) )

        High PMI = strong association
        Low PMI = weak/negative association
        """
        print("\n  Computing PMI scores...")

        total_diagnoses = sum(self.diagnosis_counts.values())
        total_concepts = sum(self.concept_counts.values())

        for dx_code in tqdm(self.diagnosis_counts.keys(), desc="  PMI"):
            p_dx = self.diagnosis_counts[dx_code] / total_diagnoses

            for concept_cui in self.concept_counts.keys():
                # Joint probability
                cooccur_count = self.diagnosis_concept_counts[dx_code].get(concept_cui, 0)
                if cooccur_count == 0:
                    continue

                p_dx_concept = cooccur_count / self.total_pairs

                # Marginal probabilities
                p_concept = self.concept_counts[concept_cui] / total_concepts

                # PMI
                pmi = math.log(p_dx_concept / (p_dx * p_concept + 1e-10) + 1e-10)

                # Store if significant
                if pmi > self.pmi_threshold:
                    key = (dx_code, concept_cui)
                    self.pmi_scores[key] = pmi

        print(f"  ✅ Computed {len(self.pmi_scores)} significant PMI scores")

    def generate_labels(self, diagnosis_codes: List[str], verbose=False) -> List[int]:
        """
        Generate concept labels for a sample based on its diagnosis codes

        Args:
            diagnosis_codes: List of ICD-10 codes for this sample

        Returns:
            Binary labels (0/1) for each concept in concept store
        """
        concept_scores = defaultdict(float)

        # For each diagnosis, get associated concepts via PMI
        for dx_code in diagnosis_codes:
            for concept_cui in self.concept_store.concepts.keys():
                key = (dx_code, concept_cui)
                if key in self.pmi_scores:
                    # Use max PMI across all diagnoses
                    concept_scores[concept_cui] = max(
                        concept_scores[concept_cui],
                        self.pmi_scores[key]
                    )

        # Convert to binary labels
        labels = []
        concept_ids = list(self.concept_store.concepts.keys())

        for cui in concept_ids:
            label = 1 if concept_scores[cui] > 0 else 0
            labels.append(label)

        if verbose:
            activated = sum(labels)
            if activated > 0:
                avg_pmi = np.mean([concept_scores[cui] for cui in concept_ids if concept_scores[cui] > 0])
                print(f"  Labels: {activated}/{len(labels)} activated (avg PMI: {avg_pmi:.3f})")

        return labels

    def generate_dataset_labels(self, df_data,
                               cache_file: str = 'diagnosis_conditional_labels.pkl') -> np.ndarray:
        """Generate labels for entire dataset with caching"""

        # Check cache
        if os.path.exists(cache_file):
            print(f"\n📦 Loading cached labels from {cache_file}...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print(f"\n🏷️  Generating diagnosis-conditional labels for {len(df_data)} samples...")

        all_labels = []
        for i, row in enumerate(tqdm(df_data.itertuples(), total=len(df_data), desc="  Labeling")):
            diagnosis_codes = row.icd_codes
            labels = self.generate_labels(diagnosis_codes, verbose=(i < 3))
            all_labels.append(labels)

        all_labels = np.array(all_labels)

        # Cache
        print(f"  💾 Caching to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(all_labels, f)

        # Stats
        avg_labels = all_labels.sum(axis=1).mean()
        print(f"  ✅ Generated labels: {all_labels.shape}")
        print(f"  📊 Avg labels per sample: {avg_labels:.1f}")

        # Distribution
        label_counts = all_labels.sum(axis=1)
        print(f"  📊 Label distribution:")
        print(f"     Min: {label_counts.min()}")
        print(f"     Median: {np.median(label_counts):.0f}")
        print(f"     Mean: {label_counts.mean():.1f}")
        print(f"     Max: {label_counts.max()}")

        return all_labels

    def _get_icd_variants(self, code: str) -> List[str]:
        """Get ICD code variants for matching"""
        variants = {code, code.replace('.', '')}
        no_dots = code.replace('.', '')
        if len(no_dots) >= 4:
            variants.add(no_dots[:3] + '.' + no_dots[3:])
        variants.add(no_dots[:3])
        return list(variants)

# ============================================================================
# SEMANTIC TYPE VALIDATOR
# ============================================================================
class SemanticTypeValidator:
    """Filters concepts by clinical relevance"""

    RELEVANT_TYPES = {
        'T047', 'T046', 'T184', 'T033', 'T048', 'T037', 'T191', 'T020',
    }

    DIAGNOSIS_SEMANTIC_GROUPS = {
        'J': {'T047', 'T046', 'T184', 'T033'},
        'I': {'T047', 'T046', 'T184', 'T033'},
        'A': {'T047', 'T046', 'T184', 'T033'},
        'K': {'T047', 'T046', 'T184', 'T033'},
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

        if diagnosis_code:
            prefix = diagnosis_code[0]
            expected_types = self.DIAGNOSIS_SEMANTIC_GROUPS.get(prefix, self.RELEVANT_TYPES)
            if not semantic_types.intersection(expected_types):
                return False

        return True

# ============================================================================
# UMLS LOADER
# ============================================================================
class FastUMLSLoader:
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

        print("  Loading MRCONSO...")
        with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  Parsing", total=max_concepts):
                if concepts_loaded >= max_concepts:
                    break

                fields = line.strip().split('|')
                if len(fields) < 15:
                    continue

                cui = fields[0]
                lang = fields[1]
                sab = fields[11]
                code = fields[13]
                term = fields[14]

                if lang != 'ENG':
                    continue
                if sab not in ['SNOMEDCT_US', 'ICD10CM', 'MSH', 'NCI']:
                    continue

                if cui not in cui_to_types:
                    continue
                types = cui_to_types[cui]
                if not any(t in target_types for t in types):
                    continue

                if cui not in self.concepts:
                    self.concepts[cui] = {
                        'cui': cui,
                        'name': term,
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
                    cui = fields[0]
                    definition = fields[5]

                    if cui in concepts and definition:
                        if 'definition' not in concepts[cui]:
                            concepts[cui]['definition'] = definition
                            definitions_added += 1

        print(f"  ✅ Added {definitions_added} definitions")
        return concepts

# ============================================================================
# MIMIC LOADER
# ============================================================================
class MIMICLoader:
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

def prepare_dataset(df_diag, df_adm, df_notes, icd_descriptions,
                   target_codes, min_samples_per_code=100):
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

    max_per_code = 3000
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

# ============================================================================
# CONCEPT STORE (300 CONCEPTS - PHASE 1 IMPROVEMENT)
# ============================================================================
class Phase4ConceptStore:
    """Phase 1: Expanded to 300 high-quality concepts (from 150)"""

    def __init__(self, umls_concepts: Dict, icd_to_cui: Dict):
        self.umls_concepts = umls_concepts
        self.icd_to_cui = icd_to_cui
        self.concepts = {}
        self.concept_to_idx = {}
        self.idx_to_concept = {}
        self.semantic_validator = SemanticTypeValidator(umls_concepts)

    def build_concept_set(self, target_icd_codes: List[str],
                         icd_descriptions: Dict[str, str],
                         target_concept_count: int = 300):
        print(f"\n🔬 Building Phase 1 concept set (target: {target_concept_count})...")

        relevant_cuis = set()

        # Strategy 1: Direct ICD mappings (increased limit)
        for icd in target_icd_codes:
            variants = self._get_icd_variants(icd)
            for variant in variants:
                if variant in self.icd_to_cui:
                    cuis = self.icd_to_cui[variant]
                    validated = [
                        cui for cui in cuis
                        if self.semantic_validator.validate_concept(cui, icd)
                    ]
                    relevant_cuis.update(validated[:60])  # Increased from 30

        print(f"  Direct mappings: {len(relevant_cuis)} concepts")

        # Strategy 2: Enhanced keyword expansion
        diagnosis_keywords = {
            'J189': ['pneumonia', 'lung infection', 'respiratory infection',
                     'infiltrate', 'bacterial pneumonia', 'aspiration',
                     'consolidation', 'atelectasis', 'pleural effusion'],
            'I5023': ['heart failure', 'cardiac failure', 'cardiomyopathy',
                      'pulmonary edema', 'ventricular dysfunction',
                      'congestive heart', 'left ventricular', 'cardiac output'],
            'A419': ['sepsis', 'septicemia', 'bacteremia', 'infection',
                     'septic shock', 'organ dysfunction', 'systemic infection',
                     'inflammatory response'],
            'K8000': ['cholecystitis', 'gallbladder', 'biliary disease',
                      'gallstone', 'cholelithiasis', 'biliary tract',
                      'cholangitis', 'bile duct']
        }

        for icd in target_icd_codes:
            keywords = diagnosis_keywords.get(icd, [])

            for cui, info in self.umls_concepts.items():
                if cui in relevant_cuis:
                    continue

                terms_text = ' '.join([info['name']] + info.get('terms', [])).lower()

                if any(kw in terms_text for kw in keywords):
                    if self.semantic_validator.validate_concept(cui, icd):
                        relevant_cuis.add(cui)

                if len(relevant_cuis) >= target_concept_count:
                    break

            if len(relevant_cuis) >= target_concept_count:
                break

        print(f"  After expansion: {len(relevant_cuis)} concepts")

        # Build final
        for cui in list(relevant_cuis)[:target_concept_count]:
            if cui in self.umls_concepts:
                concept = self.umls_concepts[cui]
                self.concepts[cui] = {
                    'cui': cui,
                    'name': concept['name'],
                    'definition': concept.get('definition', ''),
                    'terms': concept.get('terms', []),
                    'semantic_types': concept.get('semantic_types', [])
                }

        concept_list = list(self.concepts.keys())
        self.concept_to_idx = {cui: i for i, cui in enumerate(concept_list)}
        self.idx_to_concept = {i: cui for i, cui in enumerate(concept_list)}

        print(f"  ✅ Final: {len(self.concepts)} validated concepts")

        return self.concepts

    def _get_icd_variants(self, code: str) -> List[str]:
        variants = {code, code.replace('.', '')}
        no_dots = code.replace('.', '')
        if len(no_dots) >= 4:
            variants.add(no_dots[:3] + '.' + no_dots[3:])
        variants.add(no_dots[:3])
        return list(variants)

    def create_concept_embeddings(self, tokenizer, model, device):
        print("\n🧬 Creating Phase 1 concept embeddings...")

        concept_texts = []
        for cui, info in self.concepts.items():
            text = f"{info['name']}."
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

# ============================================================================
# DIAGNOSIS-AWARE RAG (Keep from Phase 4 - it worked!)
# ============================================================================
class DiagnosisAwareRAG:
    """Two-stage RAG: predict diagnosis → filter docs → retrieve"""

    def __init__(self, concept_store, umls_concepts, icd_descriptions, target_codes):
        self.concept_store = concept_store
        self.umls_concepts = umls_concepts
        self.icd_descriptions = icd_descriptions
        self.target_codes = target_codes
        self.documents = []
        self.doc_metadata = []
        self.index = None
        self.diagnosis_doc_pools = {}

    def build_document_store(self):
        print("\n📚 Building diagnosis-aware RAG store...")

        for cui, info in self.concept_store.concepts.items():
            if info.get('definition'):
                doc_text = f"{info['name']}. {info['definition']}"

                tags = set()
                name_lower = info['name'].lower()

                if any(kw in name_lower for kw in ['pneumonia', 'respiratory', 'lung', 'pulmonary']):
                    tags.add('J')
                if any(kw in name_lower for kw in ['heart', 'cardiac', 'failure', 'myocardial']):
                    tags.add('I')
                if any(kw in name_lower for kw in ['sepsis', 'infection', 'bacteremia']):
                    tags.add('A')
                if any(kw in name_lower for kw in ['gallbladder', 'biliary', 'cholecystitis']):
                    tags.add('K')

                self.documents.append(doc_text)
                self.doc_metadata.append({
                    'type': 'concept',
                    'cui': cui,
                    'name': info['name'],
                    'diagnosis_tags': tags
                })

        for icd_code in self.target_codes:
            if icd_code in self.icd_descriptions:
                desc = self.icd_descriptions[icd_code]
                self.documents.append(f"ICD-10 {icd_code}: {desc}")
                self.doc_metadata.append({
                    'type': 'icd',
                    'code': icd_code,
                    'description': desc,
                    'diagnosis_tags': {icd_code[0]}
                })

        for prefix in ['J', 'I', 'A', 'K']:
            pool_indices = [
                i for i, meta in enumerate(self.doc_metadata)
                if prefix in meta.get('diagnosis_tags', set())
            ]
            self.diagnosis_doc_pools[prefix] = pool_indices

        print(f"  ✅ Built store: {len(self.documents)} documents")
        print(f"  📊 Diagnosis pools:")
        for prefix, indices in self.diagnosis_doc_pools.items():
            print(f"     {prefix}: {len(indices)} documents")

        return self.documents

    def build_faiss_index(self, tokenizer, model, device):
        print("\n🔍 Building FAISS index...")

        batch_size = 32
        all_embeddings = []

        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(self.documents), batch_size), desc="  Indexing"):
                batch_docs = self.documents[i:i+batch_size]
                encodings = tokenizer(
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                ).to(device)

                outputs = model(**encodings)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)

        self.doc_embeddings = np.vstack(all_embeddings).astype('float32')

        dimension = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.doc_embeddings)

        print(f"  ✅ Index built: {self.index.ntotal} documents")
        return self.index

    def retrieve_with_diagnosis_filter(self, query_embeddings: np.ndarray,
                                      predicted_diagnosis_idx: int, k: int = 5):
        if self.index is None:
            raise ValueError("Index not built!")

        diagnosis_code = self.target_codes[predicted_diagnosis_idx]
        diagnosis_prefix = diagnosis_code[0]

        allowed_indices = self.diagnosis_doc_pools.get(diagnosis_prefix, list(range(len(self.documents))))

        distances, indices = self.index.search(
            query_embeddings.astype('float32'),
            min(k * 3, len(self.documents))
        )

        batch_results = []
        for query_dists, query_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(query_dists, query_indices):
                if idx in allowed_indices and idx < len(self.documents):
                    results.append((
                        self.documents[idx],
                        self.doc_metadata[idx],
                        float(dist)
                    ))

                if len(results) >= k:
                    break

            batch_results.append(results)

        return batch_results

# ============================================================================
# DATASET
# ============================================================================
class ClinicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=384,
                 concept_labels=None):
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

# ============================================================================
# MODEL (Keep from Phase 4)
# ============================================================================
class EnhancedCrossAttention(nn.Module):
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

class Phase4RevisedShifaMind(nn.Module):
    """Phase 4 Revised: With diagnosis-conditional concept labeling"""

    def __init__(self, base_model, concept_store, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.concept_store = concept_store
        self.num_classes = num_classes
        self.hidden_size = base_model.config.hidden_size
        self.fusion_layers = fusion_layers

        self.fusion_modules = nn.ModuleList([
            EnhancedCrossAttention(self.hidden_size, num_heads=8)
            for _ in fusion_layers
        ])

        self.diagnosis_head = nn.Linear(self.hidden_size, num_classes)
        self.concept_head = nn.Linear(self.hidden_size, len(concept_store.concepts))

        self.diagnosis_concept_interaction = nn.Bilinear(
            num_classes, len(concept_store.concepts), len(concept_store.concepts)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, concept_embeddings,
                return_diagnosis_only=False):
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

# ============================================================================
# PHASE 1: IMPROVED LOSS FUNCTIONS WITH FOCAL LOSS
# ============================================================================
class Phase1Loss(nn.Module):
    """Phase 1: Uses Focal Loss instead of BCE"""

    def __init__(self, stage='diagnosis', alpha=0.25, gamma=2.0):
        super().__init__()
        self.stage = stage
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, outputs, labels, concept_labels=None):
        if self.stage == 'diagnosis':
            loss = self.focal_loss(outputs['logits'], labels)
            return {
                'total': loss,
                'diagnosis': loss.item()
            }

        elif self.stage == 'concepts':
            if concept_labels is None:
                raise ValueError("concept_labels required for concept stage")

            concept_precision_loss = self.focal_loss(
                outputs['concept_scores'], concept_labels
            )

            concept_probs = torch.sigmoid(outputs['concept_scores'])
            top_k_probs = torch.topk(concept_probs, k=15, dim=1)[0]  # Increased from 12
            confidence_loss = -torch.mean(top_k_probs)

            total_loss = 0.70 * concept_precision_loss + 0.30 * confidence_loss

            return {
                'total': total_loss,
                'concept_precision': concept_precision_loss.item(),
                'confidence': confidence_loss.item(),
                'top_k_avg': top_k_probs.mean().item()
            }

        elif self.stage == 'joint':
            if concept_labels is None:
                raise ValueError("concept_labels required for joint stage")

            diagnosis_loss = self.focal_loss(outputs['logits'], labels)
            concept_precision_loss = self.focal_loss(
                outputs['concept_scores'], concept_labels
            )

            concept_probs = torch.sigmoid(outputs['concept_scores'])
            top_k_probs = torch.topk(concept_probs, k=15, dim=1)[0]  # Increased from 12
            confidence_loss = -torch.mean(top_k_probs)

            total_loss = (
                0.50 * diagnosis_loss +
                0.35 * concept_precision_loss +
                0.15 * confidence_loss
            )

            return {
                'total': total_loss,
                'diagnosis': diagnosis_loss.item(),
                'concept_precision': concept_precision_loss.item(),
                'confidence': confidence_loss.item(),
                'top_k_avg': top_k_probs.mean().item()
            }

# ============================================================================
# PHASE 1: IMPROVED TRAINER WITH MORE EPOCHS & BETTER SCHEDULING
# ============================================================================
class Phase1Trainer:
    """Phase 1: Enhanced trainer with cosine annealing and early stopping"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 concept_embeddings, diagnosis_labeler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.concept_embeddings = concept_embeddings
        self.diagnosis_labeler = diagnosis_labeler
        self.device = device

        self.history = []

    def train_stage1_diagnosis(self, epochs=5, lr=2e-5):
        """Stage 1: Increased from 3 to 5 epochs"""
        print("\n" + "="*70)
        print("STAGE 1: DIAGNOSIS HEAD TRAINING (5 EPOCHS)")
        print("="*70)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = Phase1Loss(stage='diagnosis')

        num_training_steps = epochs * len(self.train_loader)

        # PHASE 1: Cosine annealing with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )

        best_f1 = 0
        patience = 3
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            self.model.train()
            total_loss = 0
            epoch_start = time.time()

            for batch in tqdm(self.train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids, attention_mask, self.concept_embeddings,
                    return_diagnosis_only=True
                )

                loss_dict = criterion(outputs, labels)
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_loss += loss_dict['total'].item()

            avg_loss = total_loss / len(self.train_loader)
            epoch_time = time.time() - epoch_start

            val_metrics = self.evaluate(stage='diagnosis')

            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Val F1: {val_metrics['macro_f1']:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

            # Early stopping
            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']
                patience_counter = 0
                torch.save(self.model.state_dict(), 'stage1_diagnosis_phase1.pt')
                print(f"  ✅ Best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                print(f"  ⏸️  No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"  ⛔ Early stopping!")
                    break

            self.history.append({
                'stage': 'diagnosis',
                'epoch': epoch + 1,
                'loss': avg_loss,
                'val_f1': val_metrics['macro_f1']
            })

        print(f"\n✅ Stage 1 complete. Best F1: {best_f1:.4f}")
        return best_f1

    def generate_diagnosis_conditional_labels(self, df_train):
        """Stage 2: Generate diagnosis-conditional labels"""
        print("\n" + "="*70)
        print("STAGE 2: GENERATING DIAGNOSIS-CONDITIONAL LABELS")
        print("="*70)

        labels = self.diagnosis_labeler.generate_dataset_labels(
            df_train,
            cache_file='diagnosis_conditional_labels_phase1.pkl'
        )

        return labels

    def train_stage3_concepts(self, concept_labels, epochs=3, lr=2e-5):
        """Stage 3: Increased from 2 to 3 epochs"""
        print("\n" + "="*70)
        print("STAGE 3: CONCEPT HEAD TRAINING (3 EPOCHS)")
        print("="*70)

        self.train_loader.dataset.concept_labels = concept_labels

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = Phase1Loss(stage='concepts')

        num_training_steps = epochs * len(self.train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            self.model.train()
            total_loss = 0
            loss_components = defaultdict(float)
            epoch_start = time.time()

            for batch in tqdm(self.train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                concept_labels_batch = batch['concept_labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask, self.concept_embeddings)

                loss_dict = criterion(outputs, None, concept_labels_batch)
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_loss += loss_dict['total'].item()
                for key in ['concept_precision', 'confidence', 'top_k_avg']:
                    loss_components[key] += loss_dict[key]

            avg_loss = total_loss / len(self.train_loader)
            avg_top_k = loss_components['top_k_avg'] / len(self.train_loader)
            epoch_time = time.time() - epoch_start

            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Top-K: {avg_top_k:.3f}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

            torch.save(self.model.state_dict(), 'stage3_concepts_phase1.pt')

            self.history.append({
                'stage': 'concepts',
                'epoch': epoch + 1,
                'loss': avg_loss,
                'top_k': avg_top_k
            })

        print(f"\n✅ Stage 3 complete")

    def train_stage4_joint(self, concept_labels, epochs=7, lr=1.5e-5):
        """Stage 4: Increased from 3 to 7 epochs"""
        print("\n" + "="*70)
        print("STAGE 4: JOINT FINE-TUNING (7 EPOCHS)")
        print("="*70)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = Phase1Loss(stage='joint')

        num_training_steps = epochs * len(self.train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )

        best_f1 = 0
        patience = 3
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            self.model.train()
            total_loss = 0
            loss_components = defaultdict(float)
            epoch_start = time.time()

            for batch in tqdm(self.train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                concept_labels_batch = batch['concept_labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask, self.concept_embeddings)

                loss_dict = criterion(outputs, labels, concept_labels_batch)
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_loss += loss_dict['total'].item()
                for key in ['diagnosis', 'concept_precision', 'confidence', 'top_k_avg']:
                    loss_components[key] += loss_dict[key]

            avg_loss = total_loss / len(self.train_loader)
            avg_top_k = loss_components['top_k_avg'] / len(self.train_loader)
            epoch_time = time.time() - epoch_start

            val_metrics = self.evaluate(stage='joint')

            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Val F1: {val_metrics['macro_f1']:.4f}")
            print(f"  Top-K: {avg_top_k:.3f}")
            print(f"  Concepts activated: {val_metrics['avg_concepts']:.1f}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

            # Early stopping
            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']
                patience_counter = 0
                torch.save(self.model.state_dict(), 'stage4_joint_best_phase1.pt')
                print(f"  ✅ Best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                print(f"  ⏸️  No improvement ({patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"  ⛔ Early stopping!")
                    break

            self.history.append({
                'stage': 'joint',
                'epoch': epoch + 1,
                'loss': avg_loss,
                'val_f1': val_metrics['macro_f1'],
                'concepts': val_metrics['avg_concepts']
            })

        print(f"\n✅ Stage 4 complete. Best F1: {best_f1:.4f}")
        return best_f1

    def evaluate(self, stage='joint', threshold=0.5):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        all_concept_scores = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                if stage == 'diagnosis':
                    outputs = self.model(
                        input_ids, attention_mask, self.concept_embeddings,
                        return_diagnosis_only=True
                    )
                else:
                    outputs = self.model(input_ids, attention_mask, self.concept_embeddings)

                probs = torch.sigmoid(outputs['logits'])
                preds = (probs > threshold).float()

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

                if 'concept_scores' in outputs:
                    concept_scores = torch.sigmoid(outputs['concept_scores'])
                    all_concept_scores.append(concept_scores.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        all_probs = np.vstack(all_probs)

        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

        try:
            macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
        except:
            macro_auc = 0.0

        avg_concepts = 0
        if all_concept_scores:
            all_concept_scores = np.vstack(all_concept_scores)
            avg_concepts = (all_concept_scores > 0.7).sum(axis=1).mean()

        return {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'per_class_f1': per_class_f1,
            'macro_auc': macro_auc,
            'avg_concepts': avg_concepts
        }

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_final(model, dataloader, concept_embeddings, concept_labels_test,
                  device, threshold=0.7):
    """Final evaluation with concept precision metrics"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_concept_scores = []
    all_concept_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Final Eval", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, concept_embeddings)

            probs = torch.sigmoid(outputs['logits'])
            preds = (probs > 0.5).float()
            concept_scores = torch.sigmoid(outputs['concept_scores'])

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_concept_scores.append(concept_scores.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    all_concept_scores = np.vstack(all_concept_scores)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    try:
        macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
    except:
        macro_auc = 0.0

    avg_concepts = (all_concept_scores > threshold).sum(axis=1).mean()

    # Concept precision (compared to ground truth labels)
    concept_preds = (all_concept_scores > threshold).astype(int)
    concept_precision = precision_score(
        concept_labels_test, concept_preds,
        average='samples', zero_division=0
    )

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'per_class_f1': per_class_f1,
        'macro_auc': macro_auc,
        'avg_concepts': avg_concepts,
        'concept_precision': concept_precision,
        'concept_scores': all_concept_scores
    }

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_phase1_results(baseline_metrics, phase1_metrics, target_codes):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Overall metrics
    metrics = ['Macro F1', 'Micro F1', 'AUROC']
    baseline_vals = [
        baseline_metrics['macro_f1'],
        baseline_metrics['micro_f1'],
        baseline_metrics.get('macro_auc', 0)
    ]
    phase1_vals = [
        phase1_metrics['macro_f1'],
        phase1_metrics['micro_f1'],
        phase1_metrics.get('macro_auc', 0)
    ]

    x = np.arange(len(metrics))
    width = 0.35

    axes[0, 0].bar(x - width/2, baseline_vals, width, label='016.py Baseline', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x + width/2, phase1_vals, width, label='021.py Phase 1', alpha=0.8, color='green')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Overall Performance')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    # Per-class F1
    baseline_per_class = baseline_metrics['per_class_f1']
    phase1_per_class = phase1_metrics['per_class_f1']

    x = np.arange(len(target_codes))
    axes[0, 1].bar(x - width/2, baseline_per_class, width, label='016.py Baseline', alpha=0.8, color='steelblue')
    axes[0, 1].bar(x + width/2, phase1_per_class, width, label='021.py Phase 1', alpha=0.8, color='green')
    axes[0, 1].set_xlabel('ICD-10 Code')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Per-Class F1 Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(target_codes, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # Concept metrics
    concept_metrics_names = ['Avg Concepts', 'Concept\nPrecision']
    baseline_concept = [
        baseline_metrics.get('avg_concepts', 17.4) / 30,
        baseline_metrics.get('concept_precision', 0.729)
    ]
    phase1_concept = [
        phase1_metrics.get('avg_concepts', 15) / 30,
        phase1_metrics.get('concept_precision', 0.75)
    ]

    x = np.arange(len(concept_metrics_names))
    axes[1, 0].bar(x - width/2, baseline_concept, width, label='016.py Baseline', alpha=0.8, color='steelblue')
    axes[1, 0].bar(x + width/2, phase1_concept, width, label='021.py Phase 1', alpha=0.8, color='green')
    axes[1, 0].set_ylabel('Normalized Score')
    axes[1, 0].set_title('Concept Selection Quality')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(concept_metrics_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # Summary
    axes[1, 1].axis('off')
    summary_text = f"""
PHASE 1 RESULTS:

✅ Diagnosis Performance:
  F1: {baseline_metrics['macro_f1']:.4f} → {phase1_metrics['macro_f1']:.4f}
  Change: {phase1_metrics['macro_f1'] - baseline_metrics['macro_f1']:+.4f}

✅ Concept Selection:
  Activated: {baseline_metrics.get('avg_concepts', 17.4):.1f} → {phase1_metrics.get('avg_concepts', 15):.1f}
  Precision: {baseline_metrics.get('concept_precision', 0.729):.1%} → {phase1_metrics.get('concept_precision', 0.75):.1%}

🎯 Phase 1 Improvements:
  • Focal Loss (α=0.25, γ=2.0)
  • Cosine annealing + warmup
  • 8 → 15 epochs + early stopping
  • 150 → 300 concepts
"""
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center')

    plt.tight_layout()
    plt.savefig('phase1_results.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: phase1_results.png")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SHIFAMIND PHASE 1: FOCAL LOSS & TRAINING IMPROVEMENTS")
    print("="*70)

    # Load data
    print("\n📂 Loading UMLS...")
    umls_loader = FastUMLSLoader(UMLS_PATH)
    umls_concepts = umls_loader.load_concepts(max_concepts=30000)
    umls_concepts = umls_loader.load_definitions(umls_concepts)

    print("\n📂 Loading MIMIC-IV...")
    mimic_loader = MIMICLoader(MIMIC_PATH, NOTES_PATH)
    df_diagnoses = mimic_loader.load_diagnoses()
    df_admissions = mimic_loader.load_admissions()
    df_notes = mimic_loader.load_discharge_notes()

    print("\n📂 Loading ICD-10 descriptions...")
    icd10_descriptions = load_icd10_descriptions(ICD_PATH)

    TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
    print(f"\n🎯 Target diagnoses:")
    for code in TARGET_CODES:
        print(f"  {code}: {icd10_descriptions.get(code, 'Unknown')}")

    df_train, target_codes = prepare_dataset(
        df_diagnoses, df_admissions, df_notes,
        icd10_descriptions, TARGET_CODES, min_samples_per_code=100
    )

    # Build concept store (300 concepts - PHASE 1)
    print("\n" + "="*70)
    print("BUILDING CONCEPT STORE (300 CONCEPTS - PHASE 1)")
    print("="*70)

    concept_store = Phase4ConceptStore(
        umls_concepts,
        umls_loader.icd10_to_cui
    )

    concept_set = concept_store.build_concept_set(
        target_codes,
        icd10_descriptions,
        target_concept_count=300  # PHASE 1: Increased from 150
    )

    # Initialize diagnosis-conditional labeler
    print("\n" + "="*70)
    print("INITIALIZING DIAGNOSIS-CONDITIONAL LABELER")
    print("="*70)

    diagnosis_labeler = DiagnosisConditionalLabeler(
        concept_store,
        umls_loader.icd10_to_cui,
        pmi_threshold=1.0
    )

    # Build co-occurrence statistics
    diagnosis_labeler.build_cooccurrence_statistics(df_train, target_codes)

    # Load model
    print("\n" + "="*70)
    print("LOADING BIO_CLINICALBERT")
    print("="*70)

    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name).to(device)

    concept_embeddings = concept_store.create_concept_embeddings(
        tokenizer, base_model, device
    )

    # Build RAG
    print("\n" + "="*70)
    print("BUILDING DIAGNOSIS-AWARE RAG")
    print("="*70)

    rag = DiagnosisAwareRAG(
        concept_store,
        umls_concepts,
        icd10_descriptions,
        target_codes
    )
    documents = rag.build_document_store()
    rag_index = rag.build_faiss_index(tokenizer, base_model, device)

    # Split data
    print(f"\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_train['text'].values,
        np.array(df_train['labels'].tolist()),
        test_size=0.2,
        random_state=SEED
    )

    # Get matching DataFrames for label generation
    train_indices = df_train.index[df_train['text'].isin(X_train)].tolist()
    test_indices = df_train.index[df_train['text'].isin(X_test)].tolist()

    df_train_split = df_train.loc[train_indices].reset_index(drop=True)
    df_test_split = df_train.loc[test_indices].reset_index(drop=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED
    )

    val_indices = df_train_split.index[df_train_split['text'].isin(X_val)].tolist()
    train_final_indices = df_train_split.index[df_train_split['text'].isin(X_train)].tolist()

    df_train_final = df_train_split.loc[train_final_indices].reset_index(drop=True)
    df_val_split = df_train_split.loc[val_indices].reset_index(drop=True)

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Create datasets
    train_dataset = ClinicalDataset(X_train, y_train, tokenizer, max_length=384)
    val_dataset = ClinicalDataset(X_val, y_val, tokenizer, max_length=384)
    test_dataset = ClinicalDataset(X_test, y_test, tokenizer, max_length=384)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Baseline results for comparison (016.py)
    baseline_results = {
        'macro_f1': 0.7759,
        'micro_f1': 0.7734,
        'per_class_f1': np.array([0.7044, 0.8279, 0.7177, 0.8438]),
        'macro_auc': 0.87,
        'avg_concepts': 17.4,
        'concept_precision': 0.729
    }

    # Initialize model
    print("\n" + "="*70)
    print("INITIALIZING PHASE 1 MODEL")
    print("="*70)

    model = Phase4RevisedShifaMind(
        base_model=base_model,
        concept_store=concept_store,
        num_classes=len(target_codes),
        fusion_layers=[9, 11]
    ).to(device)

    print(f"  Concepts: 300 (↑ from 150)")
    print(f"  Fusion layers: 2 (Layers 9, 11)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Initialize trainer
    trainer = Phase1Trainer(
        model, train_loader, val_loader, test_loader,
        concept_embeddings, diagnosis_labeler, device
    )

    # STAGED TRAINING
    print("\n" + "="*70)
    print("PHASE 1 STAGED TRAINING (15 EPOCHS TOTAL)")
    print("="*70)

    total_start = time.time()

    # Stage 1: Diagnosis (5 epochs)
    stage1_f1 = trainer.train_stage1_diagnosis(epochs=5, lr=2e-5)

    # Stage 2: Generate diagnosis-conditional labels
    train_concept_labels = trainer.generate_diagnosis_conditional_labels(df_train_final)

    # Stage 3: Concepts (3 epochs)
    trainer.train_stage3_concepts(train_concept_labels, epochs=3, lr=2e-5)

    # Stage 4: Joint (7 epochs)
    stage4_f1 = trainer.train_stage4_joint(train_concept_labels, epochs=7, lr=1.5e-5)

    total_time = time.time() - total_start

    print(f"\n⏱️  Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Generate test concept labels
    test_concept_labels = diagnosis_labeler.generate_dataset_labels(
        df_test_split,
        cache_file='diagnosis_conditional_labels_test_phase1.pkl'
    )

    # FINAL EVALUATION
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    model.load_state_dict(torch.load('stage4_joint_best_phase1.pt'))

    final_metrics = evaluate_final(
        model, test_loader, concept_embeddings, test_concept_labels,
        device, threshold=0.7
    )

    # RESULTS
    print("\n" + "="*70)
    print("FINAL RESULTS - PHASE 1")
    print("="*70)

    print(f"\n📊 Overall Performance:")
    print(f"  016.py Baseline F1: {baseline_results['macro_f1']:.4f}")
    print(f"  021.py Phase 1 F1:  {final_metrics['macro_f1']:.4f}")

    improvement = final_metrics['macro_f1'] - baseline_results['macro_f1']
    pct = (improvement / baseline_results['macro_f1']) * 100
    print(f"  Improvement:        {improvement:+.4f} ({pct:+.1f}%)")

    print(f"\n📊 Per-Class F1 Scores:")
    for i, code in enumerate(target_codes):
        baseline_f1 = baseline_results['per_class_f1'][i]
        phase1_f1 = final_metrics['per_class_f1'][i]
        delta = phase1_f1 - baseline_f1
        print(f"  {code}: {baseline_f1:.4f} → {phase1_f1:.4f} ({delta:+.4f})")

    print(f"\n📊 Concept Selection:")
    print(f"  016.py Baseline: {baseline_results['avg_concepts']:.1f} avg (precision: {baseline_results['concept_precision']:.1%})")
    print(f"  021.py Phase 1:  {final_metrics['avg_concepts']:.1f} avg (precision: {final_metrics['concept_precision']:.1%})")

    # Visualization
    print("\n📊 Creating visualizations...")
    plot_phase1_results(baseline_results, final_metrics, target_codes)

    # Training history
    print("\n📈 Training History:")
    for entry in trainer.history:
        if entry['stage'] == 'diagnosis':
            print(f"  Stage 1 Epoch {entry['epoch']}: Loss={entry['loss']:.4f}, Val F1={entry['val_f1']:.4f}")
        elif entry['stage'] == 'concepts':
            print(f"  Stage 3 Epoch {entry['epoch']}: Loss={entry['loss']:.4f}, Top-K={entry['top_k']:.3f}")
        elif entry['stage'] == 'joint':
            print(f"  Stage 4 Epoch {entry['epoch']}: Loss={entry['loss']:.4f}, Val F1={entry['val_f1']:.4f}, Concepts={entry['concepts']:.1f}")

    # Save artifacts
    print("\n💾 Saved artifacts:")
    print("  - stage1_diagnosis_phase1.pt")
    print("  - stage3_concepts_phase1.pt")
    print("  - stage4_joint_best_phase1.pt")
    print("  - diagnosis_conditional_labels_phase1.pkl")
    print("  - phase1_results.png")

    print("\n" + "="*70)
    print("✅ PHASE 1 COMPLETE!")
    print("="*70)

    print("\n📋 Phase 1 Improvements Applied:")
    print("  ✅ Focal Loss (α=0.25, γ=2.0)")
    print("  ✅ Cosine annealing with warmup")
    print("  ✅ 8 → 15 epochs with early stopping")
    print("  ✅ 150 → 300 concepts")
    print(f"  ✅ Concept activation: {final_metrics['avg_concepts']:.1f}")
    print(f"  ✅ Concept precision: {final_metrics['concept_precision']:.1%}")
    print(f"  ✅ F1 Score: {final_metrics['macro_f1']:.4f}")

    if final_metrics['macro_f1'] >= 0.78:
        print("\n🎯 SUCCESS! F1 ≥ 0.78 achieved. Ready for Phase 2!")
    else:
        print("\n⚠️  Target F1 (0.78) not reached. Review results before Phase 2.")
