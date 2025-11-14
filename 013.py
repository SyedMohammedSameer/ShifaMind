#!/usr/bin/env python3
"""
ShifaMind Phase 2: Enhanced Concept Selection & Multi-Layer Fusion
Building on Phase 1 Success (F1: 0.7662 → Target: 0.80+)

Phase 2 Enhancements:
1. Increased concept set: 50 → 150 concepts
2. Multi-layer fusion: 1 → 2 layers (Layer 9 + 11)
3. Diagnosis-conditional concept scoring
4. Semantic type validation
5. Improved concept embeddings with definitions
6. Enhanced loss with concept precision focus
7. Sequence length: 256 → 384 tokens
8. RAG still disabled during training (Phase 3)
"""

# ============================================================================
# SETUP & IMPORTS
# ============================================================================
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Colab detection and setup
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
    get_linear_schedule_with_warmup
)
import gzip
import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
import time

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
# PHASE 2: SEMANTIC TYPE VALIDATOR
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

    # Diagnosis-specific semantic groups
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

# ============================================================================
# PHASE 2: DIAGNOSIS-CONDITIONAL RANKER
# ============================================================================
class DiagnosisConditionalRanker:
    """Ranks concepts based on diagnosis relevance"""

    def __init__(self):
        self.diagnosis_concept_cooccurrence = defaultdict(lambda: defaultdict(int))
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

        print(f"  ✅ Learned relationships for {len(self.diagnosis_counts)} diagnoses")
        print(f"     and {len(self.concept_counts)} concepts")

    def score_concepts(self, diagnosis_code: str, candidate_concepts: List[str]) -> Dict[str, float]:
        """Score concepts based on diagnosis relevance"""
        scores = {}

        for concept in candidate_concepts:
            # P(concept|diagnosis)
            p_c_given_d = 0.0
            if self.diagnosis_counts[diagnosis_code] > 0:
                p_c_given_d = (
                    self.diagnosis_concept_cooccurrence[diagnosis_code][concept] /
                    self.diagnosis_counts[diagnosis_code]
                )

            # Add smoothing for unseen combinations
            smoothing = 0.01
            scores[concept] = p_c_given_d + smoothing

        return scores

# ============================================================================
# UMLS LOADER (Same as Phase 1)
# ============================================================================
class FastUMLSLoader:
    """Optimized UMLS loader with caching"""

    def __init__(self, umls_path: Path):
        self.umls_path = umls_path
        self.concepts = {}
        self.cui_to_icd10 = defaultdict(list)
        self.icd10_to_cui = defaultdict(list)

    def load_concepts(self, max_concepts: int = 30000):
        """Load UMLS concepts - optimized version"""
        print(f"\n📚 Loading UMLS concepts (max: {max_concepts})...")

        # Target semantic types (clinically relevant only)
        target_types = {'T047', 'T046', 'T184', 'T033', 'T048', 'T037', 'T191', 'T020'}

        # Load semantic types first
        cui_to_types = self._load_semantic_types()

        # Load concepts
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

                # Fast filtering
                if lang != 'ENG':
                    continue
                if sab not in ['SNOMEDCT_US', 'ICD10CM', 'MSH', 'NCI']:
                    continue

                # Check semantic types
                if cui not in cui_to_types:
                    continue
                types = cui_to_types[cui]
                if not any(t in target_types for t in types):
                    continue

                # Add concept
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

                # ICD mappings
                if sab == 'ICD10CM' and code:
                    self.cui_to_icd10[cui].append(code)
                    self.icd10_to_cui[code].append(cui)

        print(f"  ✅ Loaded {len(self.concepts)} concepts")
        return self.concepts

    def _load_semantic_types(self) -> Dict[str, List[str]]:
        """Load semantic types mapping"""
        mrsty_path = self.umls_path / 'MRSTY.RRF'
        cui_to_types = defaultdict(list)

        with open(mrsty_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) >= 2:
                    cui_to_types[fields[0]].append(fields[1])

        return cui_to_types

    def load_definitions(self, concepts: Dict) -> Dict:
        """Load definitions - selective loading"""
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

# ============================================================================
# DATASET PREPARATION
# ============================================================================
def prepare_dataset(df_diag, df_adm, df_notes, icd_descriptions,
                   target_codes, min_samples_per_code=100):
    """Prepare balanced dataset"""
    print("\n🔧 Preparing dataset...")

    # Filter ICD-10 only
    df_diag = df_diag[df_diag['icd_version'] == 10].copy()
    df_diag['icd_code'] = df_diag['icd_code'].str.replace('.', '', regex=False)

    # Find text column
    text_col = 'text'
    if 'text' not in df_notes.columns:
        text_cols = [col for col in df_notes.columns if 'text' in col.lower()]
        text_col = text_cols[0]

    # Merge notes with diagnoses
    df_notes_with_diag = df_notes.merge(
        df_diag.groupby('hadm_id')['icd_code'].apply(list).reset_index(),
        on='hadm_id', how='inner'
    )

    df = df_notes_with_diag.rename(columns={
        'icd_code': 'icd_codes',
        text_col: 'text'
    })[['hadm_id', 'text', 'icd_codes']].copy()

    # Filter for target codes
    df['has_target'] = df['icd_codes'].apply(
        lambda codes: any(code in target_codes for code in codes)
    )
    df_filtered = df[df['has_target']].copy()

    # Create labels
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

    print(f"  ✅ Dataset: {len(df_final)} samples")

    return df_final, target_codes

# ============================================================================
# PHASE 2: ENHANCED CONCEPT STORE
# ============================================================================
class EnhancedConceptStore:
    """Enhanced concept store with semantic validation - Phase 2"""

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
        """Build enhanced concept set - Phase 2: 150 concepts with validation"""
        print(f"\n🔬 Building enhanced concept set (target: {target_concept_count})...")

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
                    relevant_cuis.update(validated[:30])  # Top 30 per code

        print(f"  Direct mappings: {len(relevant_cuis)} validated concepts")

        # Strategy 2: Diagnosis-specific keyword expansion with validation
        diagnosis_keywords = {
            'J189': ['pneumonia', 'lung infection', 'respiratory infection',
                     'consolidation', 'infiltrate', 'bacterial pneumonia',
                     'atypical pneumonia', 'aspiration', 'respiratory failure'],
            'I5023': ['heart failure', 'cardiac failure', 'ventricular failure',
                      'CHF', 'cardiomyopathy', 'cardiac dysfunction',
                      'ventricular dysfunction', 'cardiac decompensation'],
            'A419': ['sepsis', 'septicemia', 'infection', 'bacteremia',
                     'systemic infection', 'septic shock', 'SIRS',
                     'inflammatory response', 'organ dysfunction'],
            'K8000': ['cholecystitis', 'gallbladder', 'biliary disease',
                      'gallstone', 'cholelithiasis', 'biliary obstruction',
                      'choledocholithiasis']
        }

        for icd in target_icd_codes:
            keywords = diagnosis_keywords.get(icd, [])

            # Add words from ICD description
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

            if len(relevant_cuis) >= target_concept_count:
                break

        print(f"  After keyword expansion: {len(relevant_cuis)} concepts")

        # Build final concept dictionary
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

        # Build indices
        concept_list = list(self.concepts.keys())
        self.concept_to_idx = {cui: i for i, cui in enumerate(concept_list)}
        self.idx_to_concept = {i: cui for i, cui in enumerate(concept_list)}

        print(f"  ✅ Final: {len(self.concepts)} validated concepts")

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
        """Create concept embeddings with definitions - Phase 2"""
        print("\n🧬 Creating enhanced concept embeddings...")

        concept_texts = []
        for cui, info in self.concepts.items():
            # Include definition for richer embeddings
            text = f"{info['name']}."
            if info['definition']:
                text += f" {info['definition'][:150]}"  # Longer definitions
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
# PHASE 2: LIGHTWEIGHT RAG (Inference Only)
# ============================================================================
class LightweightRAG:
    """Simplified RAG for inference only"""

    def __init__(self, concept_store, umls_concepts, icd_descriptions, target_codes):
        self.concept_store = concept_store
        self.umls_concepts = umls_concepts
        self.icd_descriptions = icd_descriptions
        self.target_codes = target_codes
        self.documents = []
        self.doc_metadata = []
        self.index = None

    def build_document_store(self):
        """Build focused document store"""
        print("\n📚 Building RAG store...")

        # Add concept documents
        for cui, info in self.concept_store.concepts.items():
            if info.get('definition'):
                doc_text = f"{info['name']}. {info['definition']}"
                self.documents.append(doc_text)
                self.doc_metadata.append({
                    'type': 'concept',
                    'cui': cui,
                    'name': info['name']
                })

        # Add ICD descriptions
        for icd_code in self.target_codes:
            if icd_code in self.icd_descriptions:
                desc = self.icd_descriptions[icd_code]
                self.documents.append(f"ICD-10 {icd_code}: {desc}")
                self.doc_metadata.append({
                    'type': 'icd',
                    'code': icd_code,
                    'description': desc
                })

        print(f"  ✅ Built store: {len(self.documents)} documents")
        return self.documents

    def build_faiss_index(self, tokenizer, model, device):
        """Build FAISS index"""
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

    def retrieve(self, query_embeddings: np.ndarray, k: int = 5):
        """Simple retrieval"""
        if self.index is None:
            raise ValueError("Index not built!")

        distances, indices = self.index.search(
            query_embeddings.astype('float32'), k
        )

        batch_results = []
        for query_dists, query_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(query_dists, query_indices):
                if idx < len(self.documents):
                    results.append((
                        self.documents[idx],
                        self.doc_metadata[idx],
                        float(dist)
                    ))
            batch_results.append(results)

        return batch_results

# ============================================================================
# DATASET
# ============================================================================
class ClinicalDataset(Dataset):
    """Clinical notes dataset"""

    def __init__(self, texts, labels, tokenizer, max_length=384):
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
# PHASE 2: MULTI-LAYER FUSION MODEL
# ============================================================================
class EnhancedCrossAttention(nn.Module):
    """Enhanced cross-attention with gating - Phase 2"""

    def __init__(self, hidden_size, num_heads=6, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Gate mechanism for controlled fusion
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, concept_embeddings, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_concepts = concept_embeddings.shape[0]

        # Expand concepts for batch
        concepts_batch = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Multi-head attention
        Q = self.query(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = self.key(concepts_batch).view(
            batch_size, num_concepts, self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = self.value(concepts_batch).view(
            batch_size, num_concepts, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )

        # Gated fusion
        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        output = hidden_states + gate_values * context

        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1)

class DiagnosisConceptHead(nn.Module):
    """Diagnosis-aware concept selection head"""

    def __init__(self, hidden_size, num_concepts, num_classes):
        super().__init__()
        self.diagnosis_head = nn.Linear(hidden_size, num_classes)
        self.concept_head = nn.Linear(hidden_size, num_concepts)

        # Diagnosis-conditional concept refinement
        self.diagnosis_concept_interaction = nn.Bilinear(
            num_classes, num_concepts, num_concepts
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, cls_hidden):
        cls_hidden = self.dropout(cls_hidden)

        # Get diagnosis predictions
        diagnosis_logits = self.diagnosis_head(cls_hidden)
        diagnosis_probs = torch.sigmoid(diagnosis_logits)

        # Get concept scores
        concept_logits = self.concept_head(cls_hidden)

        # Refine concepts based on diagnosis
        refined_concept_logits = self.diagnosis_concept_interaction(
            diagnosis_probs, torch.sigmoid(concept_logits)
        )

        return diagnosis_logits, refined_concept_logits

class Phase2ShifaMind(nn.Module):
    """Phase 2: Multi-layer fusion with diagnosis-conditional concepts"""

    def __init__(self, base_model, concept_store, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.concept_store = concept_store
        self.num_classes = num_classes
        self.hidden_size = base_model.config.hidden_size
        self.fusion_layers = fusion_layers

        # Multi-layer fusion modules
        self.fusion_modules = nn.ModuleList([
            EnhancedCrossAttention(self.hidden_size, num_heads=6)
            for _ in fusion_layers
        ])

        # Diagnosis-aware prediction head
        self.prediction_head = DiagnosisConceptHead(
            self.hidden_size,
            len(concept_store.concepts),
            num_classes
        )

    def forward(self, input_ids, attention_mask, concept_embeddings):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Apply multi-layer fusion
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

            # Use last fusion output
            if i == len(self.fusion_modules) - 1:
                current_hidden = fused_hidden

        # Get CLS representation
        cls_hidden = current_hidden[:, 0, :]

        # Get predictions
        diagnosis_logits, concept_logits = self.prediction_head(cls_hidden)

        return {
            'logits': diagnosis_logits,
            'concept_scores': concept_logits,
            'attention_weights': fusion_attentions
        }

# ============================================================================
# PHASE 2: ENHANCED LOSS
# ============================================================================
class Phase2Loss(nn.Module):
    """Enhanced loss with concept precision focus - Phase 2"""

    def __init__(self, alpha=0.6, beta=0.25, gamma=0.15):
        super().__init__()
        self.alpha = alpha    # Diagnosis weight
        self.beta = beta      # Concept precision weight
        self.gamma = gamma    # Top-k concept confidence weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, diagnosis_logits, labels, concept_scores):
        # Diagnosis loss
        loss_dx = self.bce_loss(diagnosis_logits, labels)

        # Concept sparsity loss (encourage selective activation)
        concept_probs = torch.sigmoid(concept_scores)
        loss_concept_sparse = torch.mean(concept_probs)

        # Top-k concept confidence loss (encourage confident top concepts)
        top_k_probs = torch.topk(concept_probs, k=10, dim=1)[0]
        loss_concept_confidence = -torch.mean(top_k_probs)

        # Combined loss
        total_loss = (
            self.alpha * loss_dx +
            self.beta * loss_concept_sparse +
            self.gamma * loss_concept_confidence
        )

        return {
            'total': total_loss,
            'diagnosis': loss_dx.item(),
            'concept_sparse': loss_concept_sparse.item(),
            'concept_confidence': loss_concept_confidence.item(),
            'top_k_avg': top_k_probs.mean().item()
        }

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(model, dataloader, concept_embeddings, device, threshold=0.5):
    """Evaluate model"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_concept_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, concept_embeddings)

            probs = torch.sigmoid(outputs['logits'])
            preds = (probs > threshold).float()

            concept_scores = torch.sigmoid(outputs['concept_scores'])

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_concept_scores.append(concept_scores.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    all_concept_scores = np.vstack(all_concept_scores)

    # Calculate metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    try:
        macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
    except:
        macro_auc = 0.0

    # Concept metrics
    avg_concepts_activated = (all_concept_scores > 0.5).sum(axis=1).mean()

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'per_class_f1': per_class_f1,
        'macro_auc': macro_auc,
        'avg_concepts_activated': avg_concepts_activated,
        'concept_scores': all_concept_scores
    }

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_phase2_results(phase1_metrics, phase2_metrics, target_codes):
    """Plot Phase 2 comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Overall metrics
    metrics = ['Macro F1', 'Micro F1', 'AUROC']
    phase1_vals = [
        phase1_metrics['macro_f1'],
        phase1_metrics['micro_f1'],
        phase1_metrics.get('macro_auc', 0)
    ]
    phase2_vals = [
        phase2_metrics['macro_f1'],
        phase2_metrics['micro_f1'],
        phase2_metrics.get('macro_auc', 0)
    ]

    x = np.arange(len(metrics))
    width = 0.35

    axes[0].bar(x - width/2, phase1_vals, width, label='Phase 1', alpha=0.8, color='steelblue')
    axes[0].bar(x + width/2, phase2_vals, width, label='Phase 2', alpha=0.8, color='darkorange')
    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Overall Performance')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Per-class F1
    phase1_per_class = phase1_metrics['per_class_f1']
    phase2_per_class = phase2_metrics['per_class_f1']

    x = np.arange(len(target_codes))
    axes[1].bar(x - width/2, phase1_per_class, width, label='Phase 1', alpha=0.8, color='steelblue')
    axes[1].bar(x + width/2, phase2_per_class, width, label='Phase 2', alpha=0.8, color='darkorange')
    axes[1].set_xlabel('ICD-10 Code')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Per-Class F1 Score')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(target_codes, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    # Concept activation comparison
    concept_metrics = ['Avg Concepts\nActivated']
    phase1_concept = [phase1_metrics.get('avg_concepts_activated', 5)]
    phase2_concept = [phase2_metrics.get('avg_concepts_activated', 5)]

    x = np.arange(len(concept_metrics))
    axes[2].bar(x - width/2, phase1_concept, width, label='Phase 1 (50 concepts)', alpha=0.8, color='steelblue')
    axes[2].bar(x + width/2, phase2_concept, width, label='Phase 2 (150 concepts)', alpha=0.8, color='darkorange')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Concept Selection')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(concept_metrics)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase2_results.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: phase2_results.png")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SHIFAMIND PHASE 2: ENHANCED CONCEPT SELECTION")
    print("="*70)

    # ========================================================================
    # LOAD DATA
    # ========================================================================
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

    # ========================================================================
    # BUILD ENHANCED CONCEPT STORE
    # ========================================================================
    print("\n" + "="*70)
    print("BUILDING ENHANCED CONCEPT STORE (PHASE 2: 150 CONCEPTS)")
    print("="*70)

    concept_store = EnhancedConceptStore(
        umls_concepts,
        umls_loader.icd10_to_cui
    )

    concept_set = concept_store.build_concept_set(
        target_codes,
        icd10_descriptions,
        target_concept_count=150  # Phase 2: Medium set
    )

    # ========================================================================
    # LOAD MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("LOADING BIO_CLINICALBERT")
    print("="*70)

    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name).to(device)

    # Create enhanced concept embeddings
    concept_embeddings = concept_store.create_concept_embeddings(
        tokenizer, base_model, device
    )

    # ========================================================================
    # BUILD RAG (for inference only)
    # ========================================================================
    print("\n" + "="*70)
    print("BUILDING RAG (INFERENCE ONLY)")
    print("="*70)

    rag = LightweightRAG(
        concept_store,
        umls_concepts,
        icd10_descriptions,
        target_codes
    )
    documents = rag.build_document_store()
    rag_index = rag.build_faiss_index(tokenizer, base_model, device)

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

    # Create datasets (max_length=384 for Phase 2)
    train_dataset = ClinicalDataset(X_train, y_train, tokenizer, max_length=384)
    val_dataset = ClinicalDataset(X_val, y_val, tokenizer, max_length=384)
    test_dataset = ClinicalDataset(X_test, y_test, tokenizer, max_length=384)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # ========================================================================
    # LOAD PHASE 1 RESULTS (for comparison)
    # ========================================================================
    phase1_results = {
        'macro_f1': 0.7662,
        'micro_f1': 0.7455,
        'per_class_f1': np.array([0.6914, 0.8300, 0.7036, 0.8397]),
        'macro_auc': 0.85,
        'avg_concepts_activated': 5.0
    }

    # ========================================================================
    # TRAIN PHASE 2 SHIFAMIND
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING PHASE 2 SHIFAMIND")
    print("="*70)

    model = Phase2ShifaMind(
        base_model=base_model,
        concept_store=concept_store,
        num_classes=len(target_codes),
        fusion_layers=[9, 11]  # Two layers
    ).to(device)

    # Training setup
    num_epochs = 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )

    criterion = Phase2Loss(alpha=0.6, beta=0.25, gamma=0.15)

    best_f1 = 0
    training_history = []

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")

        model.train()
        total_loss = 0
        loss_components = defaultdict(float)
        epoch_start = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, concept_embeddings)

            loss_dict = criterion(
                outputs['logits'],
                labels,
                outputs['concept_scores']
            )

            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss_dict['total'].item()
            for key in ['diagnosis', 'concept_sparse', 'concept_confidence', 'top_k_avg']:
                loss_components[key] += loss_dict[key]

            progress_bar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'top_k': f"{loss_dict['top_k_avg']:.3f}"
            })

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        avg_top_k = loss_components['top_k_avg'] / len(train_loader)

        # Validation
        val_metrics = evaluate_model(
            model, val_loader, concept_embeddings, device
        )

        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Val Micro F1: {val_metrics['micro_f1']:.4f}")
        print(f"  Top-K Concept Avg: {avg_top_k:.3f}")
        print(f"  Avg Concepts Activated: {val_metrics['avg_concepts_activated']:.1f}")

        # Save best model
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), 'best_phase2_model.pt')
            print(f"  ✅ New best F1: {best_f1:.4f}")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_macro_f1': val_metrics['macro_f1'],
            'val_micro_f1': val_metrics['micro_f1'],
            'avg_concepts': val_metrics['avg_concepts_activated'],
            'time': epoch_time
        })

    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    # Load best model
    model.load_state_dict(torch.load('best_phase2_model.pt'))

    final_metrics = evaluate_model(
        model, test_loader, concept_embeddings, device
    )

    # ========================================================================
    # RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS - PHASE 2")
    print("="*70)

    print(f"\n📊 Overall Performance:")
    print(f"  Phase 1 Macro F1:  {phase1_results['macro_f1']:.4f}")
    print(f"  Phase 2 Macro F1:  {final_metrics['macro_f1']:.4f}")

    improvement = final_metrics['macro_f1'] - phase1_results['macro_f1']
    pct = (improvement / phase1_results['macro_f1']) * 100 if phase1_results['macro_f1'] > 0 else 0
    print(f"  Improvement:       {improvement:+.4f} ({pct:+.1f}%)")

    print(f"\n📊 Per-Class F1 Scores:")
    for i, code in enumerate(target_codes):
        phase1_f1 = phase1_results['per_class_f1'][i]
        phase2_f1 = final_metrics['per_class_f1'][i]
        delta = phase2_f1 - phase1_f1
        print(f"  {code}: {phase1_f1:.4f} → {phase2_f1:.4f} ({delta:+.4f})")

    print(f"\n📊 Concept Selection:")
    print(f"  Phase 1: {phase1_results['avg_concepts_activated']:.1f} avg concepts (50 total)")
    print(f"  Phase 2: {final_metrics['avg_concepts_activated']:.1f} avg concepts (150 total)")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    print("\n📊 Creating visualizations...")
    plot_phase2_results(phase1_results, final_metrics, target_codes)

    # ========================================================================
    # TRAINING SUMMARY
    # ========================================================================
    print("\n📈 Training Summary:")
    for entry in training_history:
        print(f"  Epoch {entry['epoch']}: "
              f"Loss={entry['train_loss']:.4f}, "
              f"Val F1={entry['val_macro_f1']:.4f}, "
              f"Concepts={entry['avg_concepts']:.1f}, "
              f"Time={entry['time']:.1f}s")

    total_training_time = sum(e['time'] for e in training_history)
    print(f"\n⏱️  Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")

    # ========================================================================
    # DEMO PREDICTION
    # ========================================================================
    print("\n" + "="*70)
    print("DEMO PREDICTION WITH ENHANCED CONCEPTS")
    print("="*70)

    # Get a sample case
    pneumonia_indices = df_train[
        df_train['icd_codes'].apply(lambda x: 'J189' in x)
    ].index[:1]

    if len(pneumonia_indices) > 0:
        demo_text = df_train.loc[pneumonia_indices[0], 'text']
        demo_labels = df_train.loc[pneumonia_indices[0], 'labels']

        print(f"\n📝 Sample Case:")
        print(f"  Text: {demo_text[:300]}...")

        # Encode
        encoding = tokenizer(
            demo_text,
            padding='max_length',
            truncation=True,
            max_length=384,
            return_tensors='pt'
        ).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(
                encoding['input_ids'],
                encoding['attention_mask'],
                concept_embeddings
            )

            diagnosis_probs = torch.sigmoid(outputs['logits'])[0]
            concept_scores = torch.sigmoid(outputs['concept_scores'])[0]

        print(f"\n🏥 Diagnosis Predictions:")
        for i, code in enumerate(target_codes):
            prob = diagnosis_probs[i].item()
            actual = demo_labels[i]
            status = "✅" if (prob > 0.5) == actual else "❌"
            print(f"  {status} {code}: {icd10_descriptions[code][:60]}")
            print(f"      Predicted: {prob:.1%}, Actual: {'Yes' if actual else 'No'}")

        # Top concepts
        top_k = 10
        top_scores, top_indices = torch.topk(concept_scores, k=top_k)

        print(f"\n🔬 Top {top_k} Selected Concepts (from 150):")
        concept_ids = list(concept_store.concepts.keys())
        for rank, (score, idx) in enumerate(zip(top_scores, top_indices), 1):
            cui = concept_ids[idx.item()]
            concept_info = concept_store.concepts[cui]
            print(f"  {rank}. {concept_info['name']}")
            print(f"      Score: {score.item():.3f}, CUI: {cui}")
            if rank <= 3 and concept_info['definition']:
                print(f"      Def: {concept_info['definition'][:80]}...")

        # RAG retrieval demo
        print(f"\n📚 Top 3 RAG Retrieved Documents:")
        with torch.no_grad():
            query_outputs = base_model(**encoding)
            query_emb = query_outputs.last_hidden_state[:, 0, :].cpu().numpy()

        retrieved_docs = rag.retrieve(query_emb, k=3)

        for i, (doc, metadata, dist) in enumerate(retrieved_docs[0], 1):
            print(f"\n  {i}. {metadata.get('name', metadata.get('code', 'N/A'))}")
            print(f"     {doc[:120]}...")
            print(f"     Distance: {dist:.3f}")

    # ========================================================================
    # SAVE ARTIFACTS
    # ========================================================================
    print("\n💾 Saved artifacts:")
    print("  - best_phase2_model.pt (model checkpoint)")
    print("  - phase2_results.png (performance visualization)")

    print("\n" + "="*70)
    print("✅ PHASE 2 COMPLETE!")
    print("="*70)
    print("\nPhase 2 Achievements:")
    print(f"  • Increased concepts: 50 → 150")
    print(f"  • Multi-layer fusion: 1 → 2 layers")
    print(f"  • Diagnosis-conditional concept selection")
    print(f"  • Semantic type validation")
    print(f"  • F1 Score: {phase1_results['macro_f1']:.4f} → {final_metrics['macro_f1']:.4f}")
    print("\nNext steps:")
    print("  - Phase 3: Selective RAG during training")
    print("  - Phase 4: Full system with all enhancements")
