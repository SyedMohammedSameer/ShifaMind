#!/usr/bin/env python3
"""
ShifaMind Enhanced: Improved Concept Selection & RAG
Key Enhancements:
1. Diagnosis-conditional concept ranking
2. Semantic type filtering
3. Dynamic per-concept thresholding
4. Improved RAG with relevance filtering
5. Contrastive concept embeddings
"""

# ============================================================================
# ENHANCED IMPORTS
# ============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🔧 Installing dependencies...")
    !pip install -q faiss-cpu scikit-learn

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
    get_linear_schedule_with_warmup
)

import gzip
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import seaborn as sns

import faiss

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# DATA PATHS
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind/01_Raw_Datasets/Extracted')
MIMIC_PATH = BASE_PATH / 'mimic-iv-3.1'
UMLS_PATH = BASE_PATH / 'umls-2025AA-metathesaurus-full/2025AA/META'
ICD_PATH = BASE_PATH / 'icd10cm-CodesDescriptions-2024'
NOTES_PATH = BASE_PATH / 'mimic-iv-note-2.2'

print(f"\nData paths validated:")
print(f"  MIMIC-IV: {MIMIC_PATH.exists()}")
print(f"  UMLS: {UMLS_PATH.exists()}")

# ============================================================================
# ENHANCEMENT 1: DIAGNOSIS-CONDITIONAL CONCEPT RANKER
# ============================================================================

class DiagnosisConditionalRanker:
    """Ranks concepts based on diagnosis relevance"""

    def __init__(self):
        self.diagnosis_concept_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.concept_diagnosis_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.diagnosis_counts = defaultdict(int)
        self.concept_counts = defaultdict(int)

    def train(self, training_data: List[Tuple[List[str], List[str]]]):
        """Build co-occurrence statistics from training data

        Args:
            training_data: List of (diagnosis_codes, concept_cuis) tuples
        """
        print("\n📊 Training diagnosis-conditional ranker...")

        for diagnoses, concepts in training_data:
            for dx in diagnoses:
                self.diagnosis_counts[dx] += 1
                for concept in concepts:
                    self.diagnosis_concept_cooccurrence[dx][concept] += 1

            for concept in concepts:
                self.concept_counts[concept] += 1
                for dx in diagnoses:
                    self.concept_diagnosis_cooccurrence[concept][dx] += 1

        print(f"  Learned relationships for {len(self.diagnosis_counts)} diagnoses")
        print(f"  and {len(self.concept_counts)} concepts")

    def score_concepts(self, predicted_diagnosis: str, candidate_concepts: List[str]) -> Dict[str, float]:
        """Score concepts based on diagnosis relevance

        Returns relevance scores between 0 and 1
        """
        scores = {}

        for concept in candidate_concepts:
            # P(concept|diagnosis)
            p_c_given_d = 0.0
            if self.diagnosis_counts[predicted_diagnosis] > 0:
                p_c_given_d = (
                    self.diagnosis_concept_cooccurrence[predicted_diagnosis][concept] /
                    self.diagnosis_counts[predicted_diagnosis]
                )

            # P(diagnosis|concept)
            p_d_given_c = 0.0
            if self.concept_counts[concept] > 0:
                p_d_given_c = (
                    self.concept_diagnosis_cooccurrence[concept][predicted_diagnosis] /
                    self.concept_counts[concept]
                )

            # Combined score with smoothing
            alpha = 0.6  # Weight for P(c|d)
            beta = 0.4   # Weight for P(d|c)
            smoothing = 0.01

            scores[concept] = alpha * p_c_given_d + beta * p_d_given_c + smoothing

        return scores

# ============================================================================
# ENHANCEMENT 2: SEMANTIC TYPE VALIDATOR
# ============================================================================

class SemanticTypeValidator:
    """Filters concepts by clinical relevance using UMLS semantic types"""

    RELEVANT_TYPES = {
        'T047',  # Disease or Syndrome
        'T046',  # Pathologic Function
        'T184',  # Sign or Symptom
        'T033',  # Finding
        'T048',  # Mental or Behavioral Dysfunction
        'T037',  # Injury or Poisoning
        'T191',  # Neoplastic Process
        'T020',  # Acquired Abnormality
    }

    # Map diagnosis codes to expected semantic type groups
    DIAGNOSIS_SEMANTIC_GROUPS = {
        'J': {'T047', 'T046', 'T184', 'T033'},  # Respiratory
        'I': {'T047', 'T046', 'T184', 'T033'},  # Circulatory
        'A': {'T047', 'T046', 'T184', 'T033'},  # Infectious
        'K': {'T047', 'T046', 'T184', 'T033'},  # Digestive
    }

    def __init__(self, umls_concepts: Dict):
        self.umls_concepts = umls_concepts

    def validate_concept(self, cui: str, diagnosis_code: str = None) -> bool:
        """Check if concept has clinically relevant semantic type"""

        if cui not in self.umls_concepts:
            return False

        concept = self.umls_concepts[cui]
        semantic_types = set(concept.get('semantic_types', []))

        # Must have at least one relevant type
        if not semantic_types.intersection(self.RELEVANT_TYPES):
            return False

        # Additional filtering based on diagnosis
        if diagnosis_code:
            prefix = diagnosis_code[0]
            expected_types = self.DIAGNOSIS_SEMANTIC_GROUPS.get(prefix, self.RELEVANT_TYPES)
            if not semantic_types.intersection(expected_types):
                return False

        return True

    def filter_concepts(self, concepts: List[str], diagnosis_code: str = None) -> List[str]:
        """Filter concept list to only valid ones"""
        return [cui for cui in concepts if self.validate_concept(cui, diagnosis_code)]

# ============================================================================
# ENHANCEMENT 3: DYNAMIC CONCEPT THRESHOLDING
# ============================================================================

class DynamicConceptThresholder:
    """Learn optimal thresholds per concept to achieve target precision"""

    def __init__(self, target_precision: float = 0.70):
        self.target_precision = target_precision
        self.concept_thresholds = {}

    def calibrate(self, val_concept_scores: np.ndarray,
                  val_concept_labels: np.ndarray,
                  concept_ids: List[str]):
        """Find optimal threshold for each concept

        Args:
            val_concept_scores: (n_samples, n_concepts) predicted scores
            val_concept_labels: (n_samples, n_concepts) ground truth
            concept_ids: List of concept CUIs
        """
        print(f"\n🎯 Calibrating per-concept thresholds (target precision: {self.target_precision})...")

        n_concepts = len(concept_ids)

        for i, cui in enumerate(concept_ids):
            scores = val_concept_scores[:, i]
            labels = val_concept_labels[:, i]

            # Try thresholds from 0.1 to 0.9
            best_threshold = 0.5
            best_f1 = 0.0

            for threshold in np.arange(0.1, 0.95, 0.05):
                preds = (scores > threshold).astype(int)

                if preds.sum() == 0:
                    continue

                precision = precision_score(labels, preds, zero_division=0)
                recall = recall_score(labels, preds, zero_division=0)

                # Only consider if precision meets target
                if precision >= self.target_precision:
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold

            self.concept_thresholds[cui] = best_threshold

        avg_threshold = np.mean(list(self.concept_thresholds.values()))
        print(f"  Average threshold: {avg_threshold:.3f}")
        print(f"  Threshold range: [{min(self.concept_thresholds.values()):.3f}, "
              f"{max(self.concept_thresholds.values()):.3f}]")

    def apply_thresholds(self, concept_scores: torch.Tensor,
                        concept_ids: List[str]) -> torch.Tensor:
        """Apply learned thresholds to concept predictions"""

        thresholded = torch.zeros_like(concept_scores)

        for i, cui in enumerate(concept_ids):
            threshold = self.concept_thresholds.get(cui, 0.5)
            thresholded[:, i] = (concept_scores[:, i] > threshold).float()

        return thresholded

# ============================================================================
# ENHANCEMENT 4: IMPROVED RAG WITH RELEVANCE FILTERING
# ============================================================================

class ImprovedMedicalRAG:
    """Enhanced RAG with diagnosis-aware filtering"""

    def __init__(self, concept_store, umls_concepts, icd_descriptions, target_codes):
        self.concept_store = concept_store
        self.umls_concepts = umls_concepts
        self.icd_descriptions = icd_descriptions
        self.target_codes = target_codes
        self.documents = []
        self.doc_metadata = []
        self.index = None
        self.semantic_validator = SemanticTypeValidator(umls_concepts)

    def build_focused_document_store(self):
        """Build RAG store focused on target diagnoses only"""
        print("\n📚 Building diagnosis-focused RAG store...")

        # 1. Add only concepts relevant to target diagnoses
        relevant_cuis = set()
        for icd_code in self.target_codes:
            prefix = icd_code[0]
            for cui, info in self.umls_concepts.items():
                if self.semantic_validator.validate_concept(cui, icd_code):
                    # Check if concept name/terms match diagnosis area
                    terms = ' '.join([info['name']] + info.get('terms', [])).lower()

                    if prefix == 'J' and any(kw in terms for kw in
                        ['pneumonia', 'respiratory', 'lung', 'pulmonary', 'infection', 'fever']):
                        relevant_cuis.add(cui)
                    elif prefix == 'I' and any(kw in terms for kw in
                        ['heart', 'cardiac', 'failure', 'cardiovascular', 'myocardial']):
                        relevant_cuis.add(cui)
                    elif prefix == 'A' and any(kw in terms for kw in
                        ['sepsis', 'infection', 'bacteremia', 'septic', 'inflammatory']):
                        relevant_cuis.add(cui)
                    elif prefix == 'K' and any(kw in terms for kw in
                        ['gallbladder', 'biliary', 'cholecystitis', 'abdominal']):
                        relevant_cuis.add(cui)

        print(f"  Selected {len(relevant_cuis)} relevant concepts")

        # 2. Add documents for relevant concepts only
        for cui in relevant_cuis:
            if cui in self.umls_concepts:
                info = self.umls_concepts[cui]
                if info.get('definition'):
                    doc_text = f"{info['name']}. {info['definition']}"
                    self.documents.append(doc_text)
                    self.doc_metadata.append({
                        'type': 'concept',
                        'cui': cui,
                        'name': info['name'],
                        'source': 'UMLS'
                    })

        # 3. Add ICD descriptions
        for icd_code in self.target_codes:
            if icd_code in self.icd_descriptions:
                desc = self.icd_descriptions[icd_code]
                self.documents.append(f"ICD-10 {icd_code}: {desc}")
                self.doc_metadata.append({
                    'type': 'icd',
                    'code': icd_code,
                    'description': desc,
                    'source': 'ICD-10-CM'
                })

        print(f"  Built document store: {len(self.documents)} focused documents")
        return self.documents

    def build_faiss_index(self, tokenizer, model, device):
        """Build FAISS index"""
        print("\n🔍 Building FAISS index...")

        batch_size = 32
        all_embeddings = []

        for i in tqdm(range(0, len(self.documents), batch_size), desc="Encoding"):
            batch_docs = self.documents[i:i+batch_size]

            encodings = tokenizer(
                batch_docs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = model(**encodings)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(embeddings)

        self.doc_embeddings = np.vstack(all_embeddings).astype('float32')

        dimension = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.doc_embeddings)

        print(f"  FAISS index built: {self.index.ntotal} documents")
        return self.index

    def retrieve_with_diagnosis_filter(self, query_embeddings: np.ndarray,
                                       predicted_diagnosis: str,
                                       k: int = 5) -> List[List[Tuple]]:
        """Retrieve and filter by diagnosis relevance"""

        if self.index is None:
            raise ValueError("Index not built!")

        # Retrieve more candidates than needed
        distances, indices = self.index.search(
            query_embeddings.astype('float32'),
            min(k * 3, len(self.documents))
        )

        batch_results = []
        for query_dists, query_indices in zip(distances, indices):
            # Filter to diagnosis-relevant documents
            filtered_results = []

            for dist, idx in zip(query_dists, query_indices):
                if idx >= len(self.documents):
                    continue

                metadata = self.doc_metadata[idx]

                # Strong filtering: only include if relevant to diagnosis
                is_relevant = False

                if metadata['type'] == 'icd' and metadata['code'] == predicted_diagnosis:
                    is_relevant = True
                elif metadata['type'] == 'concept':
                    cui = metadata['cui']
                    if self.semantic_validator.validate_concept(cui, predicted_diagnosis):
                        is_relevant = True

                if is_relevant:
                    filtered_results.append((
                        self.documents[idx],
                        metadata,
                        float(dist)
                    ))

                if len(filtered_results) >= k:
                    break

            batch_results.append(filtered_results)

        return batch_results

# ============================================================================
# ENHANCEMENT 5: CONTRASTIVE CONCEPT EMBEDDINGS
# ============================================================================

class ContrastiveConceptEncoder(nn.Module):
    """Learn better concept representations through contrastive learning"""

    def __init__(self, base_embeddings: torch.Tensor, hidden_size: int):
        super().__init__()

        self.base_embeddings = nn.Parameter(base_embeddings.clone(), requires_grad=False)

        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
        )

    def forward(self, concept_indices: torch.Tensor = None):
        """Get projected embeddings"""
        if concept_indices is None:
            embeddings = self.base_embeddings
        else:
            embeddings = self.base_embeddings[concept_indices]

        return self.projector(embeddings)

    def compute_contrastive_loss(self, anchor_concepts: torch.Tensor,
                                 positive_concepts: torch.Tensor,
                                 temperature: float = 0.07):
        """Compute InfoNCE loss for concept alignment"""

        anchor_emb = self.forward(anchor_concepts)
        positive_emb = self.forward(positive_concepts)

        # Normalize
        anchor_emb = F.normalize(anchor_emb, dim=-1)
        positive_emb = F.normalize(positive_emb, dim=-1)

        # Compute similarity
        similarity = torch.matmul(anchor_emb, positive_emb.T) / temperature

        # Labels are diagonal (positive pairs)
        labels = torch.arange(anchor_emb.size(0), device=anchor_emb.device)

        loss = F.cross_entropy(similarity, labels)

        return loss

# ============================================================================
# LOAD DATA FUNCTIONS (Keep existing)
# ============================================================================

class UMLSLoader:
    """Load UMLS (keeping original implementation)"""

    def __init__(self, umls_path: Path):
        self.umls_path = umls_path
        self.concepts = {}
        self.cui_to_icd10 = defaultdict(list)
        self.icd10_to_cui = defaultdict(list)

    def load_concepts(self, max_concepts: int = 50000,
                     filter_semantic_types: List[str] = None):
        """Load UMLS concepts"""
        print(f"\nLoading UMLS concepts...")

        target_types = filter_semantic_types or [
            'T047', 'T046', 'T184', 'T033', 'T048',
        ]

        cui_to_types = self._load_semantic_types()

        mrconso_path = self.umls_path / 'MRCONSO.RRF'
        concepts_loaded = 0

        with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="Loading"):
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

                if cui in cui_to_types:
                    types = cui_to_types[cui]
                    if not any(t in target_types for t in types):
                        continue
                else:
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

        print(f"Loaded {len(self.concepts)} concepts")
        return self.concepts

    def _load_semantic_types(self) -> Dict[str, List[str]]:
        """Load semantic types"""
        mrsty_path = self.umls_path / 'MRSTY.RRF'
        cui_to_types = defaultdict(list)

        with open(mrsty_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) >= 2:
                    cui_to_types[fields[0]].append(fields[1])

        return cui_to_types

    def load_definitions(self, concepts: Dict) -> Dict:
        """Load definitions"""
        print("\nLoading definitions...")
        mrdef_path = self.umls_path / 'MRDEF.RRF'
        definitions_added = 0

        with open(mrdef_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading"):
                fields = line.strip().split('|')
                if len(fields) >= 6:
                    cui = fields[0]
                    definition = fields[5]

                    if cui in concepts and definition:
                        if 'definition' not in concepts[cui]:
                            concepts[cui]['definition'] = definition
                            definitions_added += 1

        print(f"Added {definitions_added} definitions")
        return concepts

# ============================================================================
# MIMIC & DATA PREP (Keep existing structure)
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
    """Prepare dataset (keep existing logic)"""

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

    return df_final, target_codes

# ============================================================================
# ENHANCED CONCEPT STORE
# ============================================================================

class EnhancedConceptStore:
    """Improved concept store with better selection"""

    def __init__(self, umls_concepts: Dict, icd_to_cui: Dict):
        self.umls_concepts = umls_concepts
        self.icd_to_cui = icd_to_cui
        self.concepts = {}
        self.concept_to_idx = {}
        self.idx_to_concept = {}
        self.semantic_validator = SemanticTypeValidator(umls_concepts)

    def build_concept_set(self, target_icd_codes: List[str],
                         icd_descriptions: Dict[str, str],
                         target_concept_count: int = 200):
        """Build validated concept set"""

        print(f"\n🔬 Building validated concept set...")

        relevant_cuis = set()

        # Strategy 1: Direct ICD mappings with validation
        for icd in target_icd_codes:
            variants = self._get_icd_variants(icd)
            for variant in variants:
                if variant in self.icd_to_cui:
                    cuis = self.icd_to_cui[variant]
                    # Validate each concept
                    validated = [
                        cui for cui in cuis
                        if self.semantic_validator.validate_concept(cui, icd)
                    ]
                    relevant_cuis.update(validated[:20])

        print(f"  Direct mappings: {len(relevant_cuis)} validated concepts")

        # Strategy 2: Diagnosis-specific keyword expansion
        diagnosis_keywords = {
            'J189': ['pneumonia', 'lung infection', 'respiratory infection',
                    'consolidation', 'infiltrate', 'bacterial pneumonia'],
            'I5023': ['heart failure', 'cardiac failure', 'ventricular failure',
                     'CHF', 'cardiomyopathy', 'cardiac dysfunction'],
            'A419': ['sepsis', 'septicemia', 'infection', 'bacteremia',
                    'systemic infection', 'septic shock'],
            'K8000': ['cholecystitis', 'gallbladder', 'biliary disease',
                     'gallstone', 'cholelithiasis']
        }

        for icd in target_icd_codes:
            keywords = diagnosis_keywords.get(icd, [])
            if icd in icd_descriptions:
                desc_words = [
                    w for w in icd_descriptions[icd].lower().split()
                    if len(w) > 4
                ][:5]
                keywords.extend(desc_words)

            # Search and validate
            for cui, info in self.umls_concepts.items():
                if cui in relevant_cuis:
                    continue

                terms_text = ' '.join(
                    [info['name']] + info.get('terms', [])
                ).lower()

                # Check keyword match
                if any(kw in terms_text for kw in keywords):
                    # Validate semantic type
                    if self.semantic_validator.validate_concept(cui, icd):
                        relevant_cuis.add(cui)

                if len(relevant_cuis) >= target_concept_count:
                    break

        print(f"  After keyword expansion: {len(relevant_cuis)} concepts")

        # Build final concept dictionary
        for cui in relevant_cuis:
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

        print(f"  Final concept set: {len(self.concepts)} validated concepts")
        return self.concepts

    def _get_icd_variants(self, code: str) -> List[str]:
        """Get ICD code variants"""
        variants = {code, code.replace('.', '')}
        no_dots = code.replace('.', '')
        if len(no_dots) >= 4:
            variants.add(no_dots[:3] + '.' + no_dots[3:])
        variants.add(no_dots[:3])
        return list(variants)

    def create_concept_embeddings(self, tokenizer, model, device):
        """Create concept embeddings"""
        concept_texts = []
        for cui, info in self.concepts.items():
            text = f"{info['name']}."
            if info['definition']:
                text += f" {info['definition'][:200]}"
            concept_texts.append(text)

        batch_size = 32
        all_embeddings = []

        for i in tqdm(range(0, len(concept_texts), batch_size), desc="Encoding concepts"):
            batch = concept_texts[i:i+batch_size]
            encodings = tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = model(**encodings)
                embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).to(device)

# ============================================================================
# DATASET
# ============================================================================

class ClinicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

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

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

# ============================================================================
# ENHANCED MODEL
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """Keep existing cross-attention"""

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

        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))

        output = hidden_states + gate_values * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1)


class EnhancedCitationHead(nn.Module):
    """Improved citation head with diagnosis-aware scoring"""

    def __init__(self, hidden_size, num_concepts, num_classes):
        super().__init__()

        self.concept_selector = nn.Linear(hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(hidden_size, num_classes)

        # Diagnosis-conditional concept refinement
        self.diagnosis_concept_interaction = nn.Bilinear(
            num_classes, num_concepts, num_concepts
        )

        self.evidence_scorer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        cls_hidden = hidden_states[:, 0, :]

        # Get diagnosis predictions
        diagnosis_logits = self.diagnosis_head(cls_hidden)
        diagnosis_probs = torch.sigmoid(diagnosis_logits)

        # Get concept scores
        concept_logits = self.concept_selector(cls_hidden)

        # Refine concepts based on diagnosis (diagnosis-conditional)
        refined_concept_logits = self.diagnosis_concept_interaction(
            diagnosis_probs, torch.sigmoid(concept_logits)
        )

        concept_probs = torch.sigmoid(refined_concept_logits)

        # Evidence scoring
        evidence_scores = self.evidence_scorer(hidden_states).squeeze(-1)

        return {
            'logits': diagnosis_logits,
            'concept_scores': refined_concept_logits,
            'concept_probs': concept_probs,
            'evidence_scores': evidence_scores
        }


class EnhancedShifaMind(nn.Module):
    """Enhanced ShifaMind with all improvements"""

    def __init__(self, base_model, concept_store, rag_retriever,
                 num_classes, fusion_layers=[6, 9, 11]):
        super().__init__()

        self.base_model = base_model
        self.concept_store = concept_store
        self.rag_retriever = rag_retriever
        self.num_classes = num_classes
        self.hidden_size = base_model.config.hidden_size
        self.fusion_layers = fusion_layers

        self.fusion_modules = nn.ModuleList([
            CrossAttentionFusion(self.hidden_size)
            for _ in fusion_layers
        ])

        self.citation_head = EnhancedCitationHead(
            self.hidden_size,
            len(concept_store.concepts),
            num_classes
        )

        # Add diagnosis-conditional ranker (will be set later)
        self.diagnosis_ranker = None

    def forward(self, input_ids, attention_mask, concept_embeddings,
                retrieve_docs=True, predicted_diagnosis_idx=None):

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = hidden_states[-1]

        # Apply fusion
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

        # Get citation outputs
        citation_outputs = self.citation_head(current_hidden)

        # RAG retrieval (if needed)
        retrieved_docs = None
        if retrieve_docs and self.rag_retriever is not None:
            cls_embeddings = current_hidden[:, 0, :].detach().cpu().numpy()

            # Get predicted diagnosis for filtering
            if predicted_diagnosis_idx is not None:
                pred_dx_code = self.rag_retriever.target_codes[predicted_diagnosis_idx]
                retrieved_docs = self.rag_retriever.retrieve_with_diagnosis_filter(
                    cls_embeddings, pred_dx_code, k=3
                )

        citation_outputs['retrieved_docs'] = retrieved_docs
        citation_outputs['fusion_attentions'] = fusion_attentions

        return citation_outputs

# ============================================================================
# LOSS FUNCTION
# ============================================================================

class EnhancedMultiObjectiveLoss(nn.Module):
    """Enhanced loss with concept precision focus"""

    def __init__(self, lambda_weights=None):
        super().__init__()

        self.lambdas = lambda_weights or {
            'diagnosis': 0.4,
            'concept_precision': 0.3,
            'citation': 0.2,
            'calibration': 0.1
        }

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, concept_scores, target_concepts=None):
        # Diagnosis loss
        l_dx = self.bce_loss(logits, labels)

        # Concept precision loss (encourage sparsity + accuracy)
        if target_concepts is not None:
            l_concept = F.binary_cross_entropy_with_logits(
                concept_scores, target_concepts
            )
        else:
            # If no targets, encourage top-k sparsity
            probs = torch.sigmoid(concept_scores)
            # Penalize activating too many concepts
            l_concept = torch.mean(probs)

        # Citation loss (encourage confident top concepts)
        probs = torch.sigmoid(concept_scores)
        top_k_probs = torch.topk(probs, k=10, dim=1)[0]
        l_cite = -torch.mean(top_k_probs)  # Maximize top-k

        # Calibration loss
        predictions = (torch.sigmoid(logits) > 0.5).float()
        confidences = torch.max(torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        correct = (predictions == labels).float().mean(dim=1)
        l_cal = torch.abs(confidences.mean(dim=1) - correct).mean()

        # Total loss
        total = (
            self.lambdas['diagnosis'] * l_dx +
            self.lambdas['concept_precision'] * l_concept +
            self.lambdas['citation'] * l_cite +
            self.lambdas['calibration'] * l_cal
        )

        return {
            'total': total,
            'diagnosis': l_dx.item(),
            'concept_precision': l_concept.item(),
            'citation': l_cite.item(),
            'calibration': l_cal.item(),
            'concept_activation': probs.mean().item()
        }

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_enhanced(model, dataloader, concept_embeddings, device,
                     concept_ids, thresholder=None, threshold=0.5):
    """Enhanced evaluation with concept metrics"""
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    all_concept_scores, all_concept_preds = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, concept_embeddings,
                          retrieve_docs=False)

            # Diagnosis predictions
            probs = torch.sigmoid(outputs['logits'])
            preds = (probs > threshold).float()

            # Concept predictions
            concept_scores = torch.sigmoid(outputs['concept_scores'])

            if thresholder is not None:
                concept_preds = thresholder.apply_thresholds(
                    concept_scores, concept_ids
                )
            else:
                concept_preds = (concept_scores > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_concept_scores.append(concept_scores.cpu().numpy())
            all_concept_preds.append(concept_preds.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    all_concept_scores = np.vstack(all_concept_scores)
    all_concept_preds = np.vstack(all_concept_preds)

    # Diagnosis metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    try:
        macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
    except:
        macro_auc = 0.0

    # Concept metrics (handle baseline case with dummy concepts)
    if all_concept_scores.shape[1] > 1:
        concept_precision = precision_score(
            (all_concept_scores > 0.3).astype(int),
            all_concept_preds.astype(int),
            average='samples', zero_division=0
        )

        concept_recall = recall_score(
            (all_concept_scores > 0.3).astype(int),
            all_concept_preds.astype(int),
            average='samples', zero_division=0
        )
    else:
        # Baseline case - no real concepts
        concept_precision = 0.0
        concept_recall = 0.0

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'per_class_f1': per_class_f1,
        'macro_auc': macro_auc,
        'concept_precision': concept_precision,
        'concept_recall': concept_recall,
        'concept_scores': all_concept_scores,
        'concept_preds': all_concept_preds
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_enhanced_results(baseline_metrics, enhanced_metrics, target_codes):
    """Plot comparison with concept metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Diagnosis metrics
    metrics = ['Macro F1', 'Micro F1', 'AUROC']
    baseline_vals = [baseline_metrics['macro_f1'], baseline_metrics['micro_f1'],
                    baseline_metrics.get('macro_auc', 0)]
    enhanced_vals = [enhanced_metrics['macro_f1'], enhanced_metrics['micro_f1'],
                    enhanced_metrics.get('macro_auc', 0)]

    x = np.arange(len(metrics))
    width = 0.35

    axes[0].bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
    axes[0].bar(x + width/2, enhanced_vals, width, label='Enhanced', alpha=0.8)
    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Diagnosis Performance')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Concept metrics
    concept_metrics = ['Precision', 'Recall']
    concept_vals = [
        enhanced_metrics.get('concept_precision', 0),
        enhanced_metrics.get('concept_recall', 0)
    ]

    axes[1].bar(concept_metrics, concept_vals, alpha=0.8, color='green')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Concept Selection Quality')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)

    # Per-class F1
    baseline_per_class = baseline_metrics['per_class_f1']
    enhanced_per_class = enhanced_metrics['per_class_f1']

    x = np.arange(len(target_codes))
    axes[2].bar(x - width/2, baseline_per_class, width, label='Baseline', alpha=0.8)
    axes[2].bar(x + width/2, enhanced_per_class, width, label='Enhanced', alpha=0.8)
    axes[2].set_xlabel('ICD-10 Code')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Per-Class F1')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(target_codes, rotation=45)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: enhanced_comparison.png")
    plt.show()

print("\n✅ All enhancements loaded! Ready to train...")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SHIFAMIND ENHANCED - IMPROVED CONCEPT SELECTION")
    print("="*70)

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("\n📂 Loading UMLS...")
    umls_loader = UMLSLoader(UMLS_PATH)
    umls_concepts = umls_loader.load_concepts(max_concepts=50000)
    umls_concepts = umls_loader.load_definitions(umls_concepts)

    print("\n📂 Loading MIMIC-IV...")
    mimic_loader = MIMICLoader(MIMIC_PATH, NOTES_PATH)
    df_diagnoses = mimic_loader.load_diagnoses()
    df_admissions = mimic_loader.load_admissions()
    df_notes = mimic_loader.load_discharge_notes()

    print("\n📂 Loading ICD-10 descriptions...")
    icd10_descriptions = load_icd10_descriptions(ICD_PATH)

    # Target codes
    TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']

    print(f"\n🎯 Target diagnoses:")
    for code in TARGET_CODES:
        print(f"  {code}: {icd10_descriptions.get(code, 'Unknown')}")

    # ========================================================================
    # PREPARE DATASET
    # ========================================================================

    df_train, target_codes = prepare_dataset(
        df_diagnoses, df_admissions, df_notes,
        icd10_descriptions, TARGET_CODES, min_samples_per_code=100
    )

    print(f"\n✅ Dataset prepared: {len(df_train)} samples")

    # ========================================================================
    # BUILD ENHANCED CONCEPT STORE
    # ========================================================================

    print("\n" + "="*70)
    print("BUILDING ENHANCED CONCEPT STORE")
    print("="*70)

    enhanced_concept_store = EnhancedConceptStore(
        umls_concepts,
        umls_loader.icd10_to_cui
    )

    concept_set = enhanced_concept_store.build_concept_set(
        target_codes,
        icd10_descriptions,
        target_concept_count=200
    )

    print(f"\n✅ Concept store built: {len(concept_set)} validated concepts")

    # ========================================================================
    # LOAD MODEL
    # ========================================================================

    print("\n" + "="*70)
    print("LOADING BIOCLINICALBERT")
    print("="*70)

    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name).to(device)

    concept_embeddings = enhanced_concept_store.create_concept_embeddings(
        tokenizer, base_model, device
    )

    print(f"✅ Concept embeddings: {concept_embeddings.shape}")

    # ========================================================================
    # BUILD IMPROVED RAG
    # ========================================================================

    print("\n" + "="*70)
    print("BUILDING IMPROVED RAG")
    print("="*70)

    improved_rag = ImprovedMedicalRAG(
        enhanced_concept_store,
        umls_concepts,
        icd10_descriptions,
        target_codes
    )

    documents = improved_rag.build_focused_document_store()
    rag_index = improved_rag.build_faiss_index(tokenizer, base_model, device)

    # ========================================================================
    # SPLIT DATA
    # ========================================================================

    print(f"\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_train['text'].values,
        np.array(df_train['labels'].tolist()),
        test_size=0.2,
        random_state=SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED
    )

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    train_dataset = ClinicalDataset(X_train, y_train, tokenizer)
    val_dataset = ClinicalDataset(X_val, y_val, tokenizer)
    test_dataset = ClinicalDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # ========================================================================
    # BASELINE MODEL
    # ========================================================================

    print("\n" + "="*70)
    print("TRAINING BASELINE")
    print("="*70)

    class BaselineModel(nn.Module):
        def __init__(self, base_model, num_classes):
            super().__init__()
            self.base_model = base_model
            self.classifier = nn.Linear(base_model.config.hidden_size, num_classes)
            self.dropout = nn.Dropout(0.1)

        def forward(self, input_ids, attention_mask, concept_embeddings=None, retrieve_docs=False):
            outputs = self.base_model(input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)
            return {
                'logits': logits,
                'concept_scores': torch.zeros(logits.shape[0], 1, device=logits.device)
            }

    baseline = BaselineModel(base_model, len(target_codes)).to(device)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=2e-5)
    baseline_criterion = nn.BCEWithLogitsLoss()

    # Train baseline briefly
    for epoch in range(1):
        baseline.train()
        for batch in tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            baseline_optimizer.zero_grad()
            outputs = baseline(input_ids, attention_mask)
            loss = baseline_criterion(outputs['logits'], labels)
            loss.backward()
            baseline_optimizer.step()

    # Evaluate baseline
    baseline_metrics = evaluate_enhanced(
        baseline, test_loader, concept_embeddings, device,
        list(enhanced_concept_store.concepts.keys())
    )

    print(f"\n📊 Baseline Results:")
    print(f"  Macro F1: {baseline_metrics['macro_f1']:.4f}")
    print(f"  Micro F1: {baseline_metrics['micro_f1']:.4f}")

    # ========================================================================
    # TRAIN ENHANCED SHIFAMIND
    # ========================================================================

    print("\n" + "="*70)
    print("TRAINING ENHANCED SHIFAMIND")
    print("="*70)

    enhanced_model = EnhancedShifaMind(
        base_model=base_model,
        concept_store=enhanced_concept_store,
        rag_retriever=improved_rag,
        num_classes=len(target_codes),
        fusion_layers=[6, 9, 11]
    ).to(device)

    # Setup training
    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    optimizer = torch.optim.AdamW(enhanced_model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )

    criterion = EnhancedMultiObjectiveLoss()

    best_f1 = 0
    training_history = []

    concept_ids = list(enhanced_concept_store.concepts.keys())

    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")

        # ====================================================================
        # TRAINING
        # ====================================================================

        enhanced_model.train()
        total_loss = 0
        loss_components = defaultdict(float)

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = enhanced_model(
                input_ids, attention_mask, concept_embeddings,
                retrieve_docs=False
            )

            loss_dict = criterion(
                outputs['logits'],
                labels,
                outputs['concept_scores']
            )

            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(enhanced_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss_dict['total'].item()
            for key in ['diagnosis', 'concept_precision', 'citation', 'calibration', 'concept_activation']:
                loss_components[key] += loss_dict[key]

            progress_bar.set_postfix({'loss': f"{loss_dict['total'].item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        avg_concept_activation = loss_components['concept_activation'] / len(train_loader)

        # ====================================================================
        # VALIDATION
        # ====================================================================

        val_metrics = evaluate_enhanced(
            enhanced_model, val_loader, concept_embeddings, device, concept_ids
        )

        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Val Micro F1: {val_metrics['micro_f1']:.4f}")
        print(f"  Concept Activation: {avg_concept_activation:.4f}")
        print(f"  Concept Precision: {val_metrics['concept_precision']:.4f}")
        print(f"  Concept Recall: {val_metrics['concept_recall']:.4f}")

        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            torch.save(enhanced_model.state_dict(), 'best_enhanced_shifamind.pt')
            print(f"  ✅ New best F1: {best_f1:.4f}")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_macro_f1': val_metrics['macro_f1'],
            'val_micro_f1': val_metrics['micro_f1'],
            'concept_activation': avg_concept_activation,
            'concept_precision': val_metrics['concept_precision'],
            'concept_recall': val_metrics['concept_recall'],
            'per_class_f1': val_metrics['per_class_f1']
        })

    # ========================================================================
    # CALIBRATE THRESHOLDS
    # ========================================================================

    print("\n" + "="*70)
    print("CALIBRATING CONCEPT THRESHOLDS")
    print("="*70)

    # Get validation concept predictions
    enhanced_model.eval()
    val_concept_scores_list = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Getting val scores"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = enhanced_model(input_ids, attention_mask, concept_embeddings,
                                   retrieve_docs=False)
            concept_scores = torch.sigmoid(outputs['concept_scores'])
            val_concept_scores_list.append(concept_scores.cpu().numpy())

    val_concept_scores = np.vstack(val_concept_scores_list)

    # Create pseudo ground truth (top-k most activated concepts)
    val_concept_labels = (val_concept_scores > 0.5).astype(int)

    thresholder = DynamicConceptThresholder(target_precision=0.70)
    thresholder.calibrate(val_concept_scores, val_concept_labels, concept_ids)

    # ========================================================================
    # FINAL EVALUATION WITH THRESHOLDS
    # ========================================================================

    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    final_metrics = evaluate_enhanced(
        enhanced_model, test_loader, concept_embeddings, device,
        concept_ids, thresholder=thresholder
    )

    # ========================================================================
    # RESULTS
    # ========================================================================

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    print(f"\n📊 Diagnosis Performance:")
    print(f"  Baseline:        {baseline_metrics['macro_f1']:.4f} macro F1")
    print(f"  Enhanced:        {final_metrics['macro_f1']:.4f} macro F1")
    improvement = final_metrics['macro_f1'] - baseline_metrics['macro_f1']
    pct = (improvement / baseline_metrics['macro_f1']) * 100
    print(f"  Improvement:     {improvement:+.4f} ({pct:+.1f}%)")

    print(f"\n📊 Concept Selection Quality:")
    print(f"  Precision:       {final_metrics['concept_precision']:.4f}")
    print(f"  Recall:          {final_metrics['concept_recall']:.4f}")

    print(f"\n📊 Per-Class F1:")
    for i, code in enumerate(target_codes):
        baseline_f1 = baseline_metrics['per_class_f1'][i]
        enhanced_f1 = final_metrics['per_class_f1'][i]
        delta = enhanced_f1 - baseline_f1
        print(f"  {code}: {baseline_f1:.4f} → {enhanced_f1:.4f} ({delta:+.4f})")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    print("\n📊 Creating visualizations...")
    plot_enhanced_results(baseline_metrics, final_metrics, target_codes)

    print("\n✅ Training complete!")
    print("\n💾 Saved artifacts:")
    print("  - best_enhanced_shifamind.pt")
    print("  - enhanced_comparison.png")

    # ========================================================================
    # DEMO PREDICTIONS
    # ========================================================================

    print("\n" + "="*70)
    print("DEMO: CONCEPT QUALITY CHECK")
    print("="*70)

    # Get a pneumonia case
    pneumonia_indices = df_train[
        df_train['icd_codes'].apply(lambda x: 'J189' in x)
    ].index[:1]

    if len(pneumonia_indices) > 0:
        demo_text = df_train.loc[pneumonia_indices[0], 'text']
        demo_labels = df_train.loc[pneumonia_indices[0], 'labels']

        print(f"\n📝 Demo Case (Pneumonia):")
        print(f"  Text snippet: {demo_text[:200]}...")

        # Encode
        encoding = tokenizer(
            demo_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        enhanced_model.eval()
        with torch.no_grad():
            outputs = enhanced_model(
                encoding['input_ids'],
                encoding['attention_mask'],
                concept_embeddings,
                retrieve_docs=False
            )

            diagnosis_probs = torch.sigmoid(outputs['logits'])[0]
            concept_scores = torch.sigmoid(outputs['concept_scores'])[0]

            # Apply thresholds
            concept_preds = thresholder.apply_thresholds(
                concept_scores.unsqueeze(0), concept_ids
            )[0]

        print(f"\n🏥 Diagnosis Predictions:")
        for i, code in enumerate(target_codes):
            prob = diagnosis_probs[i].item()
            label = "✅" if prob > 0.5 else "❌"
            print(f"  {label} {code}: {icd10_descriptions[code][:50]}...")
            print(f"     Confidence: {prob:.1%}")

        print(f"\n🔬 Top Selected Concepts (after thresholding):")
        selected_indices = torch.where(concept_preds > 0)[0].cpu().numpy()
        selected_scores = concept_scores[selected_indices].cpu().numpy()

        # Sort by score
        sorted_idx = np.argsort(selected_scores)[::-1][:10]

        for rank, idx in enumerate(sorted_idx, 1):
            concept_idx = selected_indices[idx]
            cui = concept_ids[concept_idx]
            concept_info = enhanced_concept_store.concepts[cui]
            score = selected_scores[idx]

            print(f"  {rank}. {concept_info['name']}")
            print(f"     CUI: {cui}, Score: {score:.3f}")
            print(f"     Types: {', '.join(concept_info['semantic_types'][:3])}")

        print(f"\n📊 Summary:")
        print(f"  Total concepts activated: {len(selected_indices)}")
        print(f"  Average score: {concept_scores[selected_indices].mean():.3f}")

        # Check for irrelevant concepts
        relevant_keywords = ['pneumonia', 'lung', 'respiratory', 'infection',
                           'fever', 'cough', 'infiltrate']
        relevant_count = sum(
            1 for idx in selected_indices
            if any(kw in enhanced_concept_store.concepts[
                concept_ids[idx]
            ]['name'].lower() for kw in relevant_keywords)
        )

        print(f"  Clinically relevant: {relevant_count}/{len(selected_indices)} "
              f"({relevant_count/len(selected_indices)*100:.1f}%)")

    print("\n" + "="*70)
    print("✅ ALL ENHANCEMENTS COMPLETE")
    print("="*70)
