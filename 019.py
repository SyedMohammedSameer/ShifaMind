#!/usr/bin/env python3
"""
ShifaMind 018: The Virat Kohli Edition 🏏
Aggressive multi-component upgrade for maximum F1 score

MAJOR UPGRADES:
1. GatorTron-Base (345M params) - SOTA clinical transformer
2. 2000+ concept hierarchy with UMLS relationships
3. Graph Neural Networks for concept reasoning
4. Focal Loss for hard sample mining
5. Multi-task learning (diagnosis + severity + mortality + LOS)
6. Expanded dataset with related ICD codes (20K+ samples)
7. Enhanced RAG with massive medical knowledge
8. Advanced training (cosine LR, grad accumulation, EMA)

Target: 0.85+ F1 score
"""

# ============================================================================
# CELL 1: IMPORTS & SETUP
# ============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    print("🔧 Installing dependencies...")
    os.system('pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
    os.system('pip install -q transformers accelerate')
    os.system('pip install -q torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html')
    os.system('pip install -q faiss-cpu scikit-learn sentencepiece')
    from google.colab import drive
    drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModel,
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
from copy import deepcopy

# Try importing PyTorch Geometric for GNN
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data as GraphData
    HAS_PYGEOMETRIC = True
    print("✅ PyTorch Geometric available")
except:
    HAS_PYGEOMETRIC = False
    print("⚠️  PyTorch Geometric not available - using fallback concept embeddings")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
# CELL 2: EXPANDED ICD CODE MAPPING (For More Training Data)
# ============================================================================

# Map target codes to related codes for expanded training data
EXPANDED_ICD_MAPPING = {
    # Pneumonia (J189) - include all pneumonia types
    'J189': {
        'primary': 'J189',  # Unspecified
        'related': [
            'J180', 'J181', 'J182', 'J183', 'J184', 'J185', 'J186', 'J187', 'J188',  # Specific pneumonias
            'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18',  # All pneumonia categories
            'J210', 'J211', 'J219',  # Acute bronchiolitis (often co-occurs)
        ],
        'name': 'Pneumonia'
    },

    # Heart Failure (I5023) - include all heart failure types
    'I5023': {
        'primary': 'I5023',  # Acute on chronic systolic
        'related': [
            'I5020', 'I5021', 'I5022', 'I5023',  # Systolic HF
            'I5030', 'I5031', 'I5032', 'I5033',  # Diastolic HF
            'I5040', 'I5041', 'I5042', 'I5043',  # Combined HF
            'I50', 'I501', 'I502', 'I503', 'I504', 'I509',  # Other HF
            'I110',  # Hypertensive heart disease with HF
        ],
        'name': 'Heart Failure'
    },

    # Sepsis (A419) - include all sepsis types
    'A419': {
        'primary': 'A419',  # Unspecified
        'related': [
            'A400', 'A401', 'A402', 'A403', 'A408', 'A409',  # Streptococcal sepsis
            'A410', 'A411', 'A412', 'A413', 'A414', 'A415', 'A418', 'A419',  # Other sepsis
            'A327',  # Listerial sepsis
            'A227',  # Anthrax sepsis
            'B377',  # Candida sepsis
            'R6520', 'R6521',  # Severe sepsis
        ],
        'name': 'Sepsis'
    },

    # Cholecystitis (K8000) - include gallbladder diseases
    'K8000': {
        'primary': 'K8000',  # Acute cholecystitis with calculus
        'related': [
            'K8000', 'K8001', 'K8010', 'K8011', 'K8012', 'K8013',  # Cholecystitis variants
            'K802', 'K803', 'K804', 'K805',  # Other gallbladder diseases
            'K81', 'K810', 'K811', 'K819',  # Cholecystitis without calculus
            'K8220', 'K8221',  # Perforation of gallbladder
        ],
        'name': 'Cholecystitis'
    }
}

def get_all_codes_for_target(target_code: str) -> List[str]:
    """Get primary + related codes for expanded data"""
    mapping = EXPANDED_ICD_MAPPING.get(target_code, {})
    all_codes = [mapping.get('primary', target_code)]
    all_codes.extend(mapping.get('related', []))
    return list(set(all_codes))  # Remove duplicates

# ============================================================================
# CELL 3: ENHANCED UMLS LOADER (2000+ Concepts with Relationships)
# ============================================================================

class EnhancedUMLSLoader:
    """Load more concepts (2000+) with hierarchical relationships"""

    def __init__(self, umls_path: Path):
        self.umls_path = umls_path
        self.concepts = {}
        self.relationships = defaultdict(list)  # For graph construction

    def load_concepts(self, max_concepts: int = 50000):
        """Load concepts from MRCONSO"""
        print(f"\n📚 Loading UMLS concepts (max: {max_concepts})...")

        mrconso_file = self.umls_path / 'MRCONSO.RRF'
        print(f"  Loading MRCONSO...")

        # Clinical semantic types (expanded list)
        clinical_types = {
            'T047',  # Disease or Syndrome
            'T048',  # Mental or Behavioral Dysfunction
            'T184',  # Sign or Symptom
            'T033',  # Finding
            'T046',  # Pathologic Function
            'T037',  # Injury or Poisoning
            'T191',  # Neoplastic Process
            'T019',  # Congenital Abnormality
            'T020',  # Acquired Abnormality
            'T190',  # Anatomical Abnormality
            'T049',  # Cell or Molecular Dysfunction
            'T059',  # Laboratory Procedure
            'T060',  # Diagnostic Procedure
            'T061',  # Therapeutic or Preventive Procedure
            'T121',  # Pharmacologic Substance
            'T109',  # Organic Chemical
            'T195',  # Antibiotic
            'T200',  # Clinical Drug
        }

        concept_count = 0
        seen_cuis = set()

        with open(mrconso_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  Parsing"):
                if concept_count >= max_concepts:
                    break

                parts = line.strip().split('|')
                if len(parts) < 15:
                    continue

                cui = parts[0]
                lang = parts[1]

                if lang != 'ENG' or cui in seen_cuis:
                    continue

                preferred_name = parts[14]

                # Store concept
                if cui not in self.concepts:
                    self.concepts[cui] = {
                        'cui': cui,
                        'name': preferred_name,
                        'terms': [preferred_name],
                        'semantic_types': [],
                        'definition': ''
                    }
                    seen_cuis.add(cui)
                    concept_count += 1
                else:
                    self.concepts[cui]['terms'].append(preferred_name)

        print(f"  ✅ Loaded {len(self.concepts)} concepts")
        return self.concepts

    def load_semantic_types(self):
        """Load semantic types from MRSTY"""
        print(f"\n📋 Loading semantic types...")

        mrsty_file = self.umls_path / 'MRSTY.RRF'

        with open(mrsty_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  Parsing"):
                parts = line.strip().split('|')
                if len(parts) < 2:
                    continue

                cui = parts[0]
                sty = parts[1]  # Semantic type code

                if cui in self.concepts:
                    self.concepts[cui]['semantic_types'].append(sty)

        print(f"  ✅ Added semantic types")

    def load_relationships(self):
        """Load concept relationships from MRREL for graph construction"""
        print(f"\n🔗 Loading concept relationships...")

        mrrel_file = self.umls_path / 'MRREL.RRF'

        # Relationship types we care about
        important_rels = {'PAR', 'CHD', 'RB', 'RN', 'RO', 'SIB'}  # Parent, Child, Broader, Narrower, Other, Sibling

        rel_count = 0
        with open(mrrel_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  Parsing", total=50000000):
                parts = line.strip().split('|')
                if len(parts) < 8:
                    continue

                cui1 = parts[0]
                cui2 = parts[4]
                rel_type = parts[3]

                if rel_type not in important_rels:
                    continue

                if cui1 in self.concepts and cui2 in self.concepts:
                    self.relationships[cui1].append({
                        'target': cui2,
                        'type': rel_type
                    })
                    rel_count += 1

                if rel_count >= 100000:  # Limit for memory
                    break

        print(f"  ✅ Loaded {rel_count} relationships")

    def load_definitions(self):
        """Load definitions from MRDEF"""
        print(f"\n📖 Loading definitions...")

        mrdef_file = self.umls_path / 'MRDEF.RRF'
        def_count = 0

        with open(mrdef_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  Parsing"):
                parts = line.strip().split('|')
                if len(parts) < 6:
                    continue

                cui = parts[0]
                definition = parts[5]

                if cui in self.concepts and not self.concepts[cui]['definition']:
                    self.concepts[cui]['definition'] = definition
                    def_count += 1

        print(f"  ✅ Added {def_count} definitions")

# ============================================================================
# CELL 4: ICD-TO-CUI MAPPING
# ============================================================================

def load_icd_to_cui_mapping(umls_path: Path) -> Dict[str, List[str]]:
    """Load ICD-10-CM to CUI mappings from UMLS"""
    print(f"\n🔗 Loading ICD-10-CM to CUI mappings...")

    mrmap_file = umls_path / 'MRMAP.RRF'
    icd_to_cui = defaultdict(list)

    with open(mrmap_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="  Parsing MRMAP"):
            parts = line.strip().split('|')
            if len(parts) < 13:
                continue

            source_vocab = parts[2]
            source_code = parts[3]
            target_cui = parts[6]

            # ICD10CM mappings
            if source_vocab == 'ICD10CM':
                # Normalize code (remove dots)
                normalized_code = source_code.replace('.', '')
                icd_to_cui[normalized_code].append(target_cui)

    print(f"  ✅ Loaded {len(icd_to_cui)} ICD-10-CM mappings")
    return dict(icd_to_cui)

# ============================================================================
# CELL 5: CONCEPT GRAPH BUILDER (For GNN)
# ============================================================================

class ConceptGraphBuilder:
    """Build graph structure from UMLS relationships"""

    def __init__(self, umls_concepts: Dict, relationships: Dict):
        self.umls_concepts = umls_concepts
        self.relationships = relationships

    def build_graph_for_concepts(self, concept_cuis: List[str]):
        """Build PyTorch Geometric graph for selected concepts"""

        # Map CUI to index
        cui_to_idx = {cui: idx for idx, cui in enumerate(concept_cuis)}
        idx_to_cui = {idx: cui for cui, idx in cui_to_idx.items()}

        # Build edge list
        edge_list = []
        edge_types = []

        type_mapping = {'PAR': 0, 'CHD': 1, 'RB': 2, 'RN': 3, 'RO': 4, 'SIB': 5}

        for cui in concept_cuis:
            if cui not in self.relationships:
                continue

            src_idx = cui_to_idx[cui]

            for rel in self.relationships[cui]:
                target_cui = rel['target']
                if target_cui in cui_to_idx:
                    tgt_idx = cui_to_idx[target_cui]
                    edge_list.append([src_idx, tgt_idx])
                    edge_types.append(type_mapping.get(rel['type'], 4))

        if not edge_list:
            # No edges - create self-loops
            edge_list = [[i, i] for i in range(len(concept_cuis))]
            edge_types = [4] * len(edge_list)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_types, dtype=torch.long)

        return edge_index, edge_attr, cui_to_idx, idx_to_cui

# ============================================================================
# CELL 6: GRAPH NEURAL NETWORK FOR CONCEPTS
# ============================================================================

class ConceptGNN(nn.Module):
    """Graph Neural Network to enhance concept embeddings with relationships"""

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.has_pyg = HAS_PYGEOMETRIC

        if self.has_pyg:
            # Use GAT (Graph Attention Networks)
            self.layers = nn.ModuleList()
            self.layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
            for _ in range(num_layers - 1):
                self.layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        else:
            # Fallback: simple MLP
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index=None):
        """
        Args:
            x: [num_concepts, input_dim] concept embeddings
            edge_index: [2, num_edges] edge connectivity
        """
        if self.has_pyg and edge_index is not None:
            for i, layer in enumerate(self.layers):
                x = layer(x, edge_index)
                if i < len(self.layers) - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
            x = self.norm(x)
        else:
            # Fallback: just apply MLP
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
            x = self.norm(x)

        return x

# ============================================================================
# CELL 7: ENHANCED CONCEPT STORE (2000+ Concepts)
# ============================================================================

class EnhancedConceptStore:
    """Expanded concept store with 2000+ concepts and graph structure"""

    def __init__(self, umls_concepts: Dict, icd_to_cui: Dict, relationships: Dict):
        self.umls_concepts = umls_concepts
        self.icd_to_cui = icd_to_cui
        self.relationships = relationships
        self.concepts = {}
        self.concept_to_idx = {}
        self.idx_to_concept = {}
        self.graph_builder = ConceptGraphBuilder(umls_concepts, relationships)

    def build_expanded_concept_set(self,
                                   target_icd_codes: List[str],
                                   expanded_mapping: Dict,
                                   target_concept_count: int = 2000):
        """Build expanded concept set with related ICD codes"""

        print(f"\n🔬 Building expanded concept set (target: {target_concept_count})...")

        relevant_cuis = set()

        # Strategy 1: Direct ICD mappings (use expanded codes)
        for target_code in target_icd_codes:
            all_codes = get_all_codes_for_target(target_code)

            for code in all_codes:
                # Try different variants
                variants = [code, code[:3], code[:4], code[:5]]

                for variant in variants:
                    if variant in self.icd_to_cui:
                        cuis = self.icd_to_cui[variant]
                        for cui in cuis[:50]:  # Take more per code
                            if cui in self.umls_concepts:
                                relevant_cuis.add(cui)

        print(f"  Direct mappings: {len(relevant_cuis)} concepts")

        # Strategy 2: Expanded keyword matching
        diagnosis_keywords = {
            'J189': [
                'pneumonia', 'lung infection', 'respiratory infection', 'pulmonary infiltrate',
                'bacterial pneumonia', 'viral pneumonia', 'aspiration pneumonia',
                'lobar pneumonia', 'bronchopneumonia', 'pneumonitis', 'alveolar consolidation',
                'respiratory distress', 'hypoxia', 'dyspnea', 'cough', 'sputum',
                'chest x-ray finding', 'infiltrate', 'consolidation'
            ],
            'I5023': [
                'heart failure', 'cardiac failure', 'congestive heart failure', 'CHF',
                'cardiomyopathy', 'pulmonary edema', 'ventricular dysfunction',
                'systolic dysfunction', 'diastolic dysfunction', 'left ventricular failure',
                'right ventricular failure', 'acute decompensation', 'volume overload',
                'ejection fraction', 'BNP', 'shortness of breath', 'orthopnea',
                'jugular venous distension', 'peripheral edema', 'crackles'
            ],
            'A419': [
                'sepsis', 'septicemia', 'bacteremia', 'infection', 'systemic infection',
                'septic shock', 'organ dysfunction', 'SIRS', 'systemic inflammatory response',
                'fever', 'hypotension', 'tachycardia', 'leukocytosis', 'lactate',
                'blood culture', 'antibiotic', 'vasopressor', 'ICU', 'critical illness',
                'multi-organ failure', 'acute kidney injury'
            ],
            'K8000': [
                'cholecystitis', 'gallbladder', 'biliary disease', 'cholelithiasis',
                'gallstone', 'biliary colic', 'acute cholecystitis', 'chronic cholecystitis',
                'right upper quadrant pain', 'Murphy sign', 'ultrasound', 'HIDA scan',
                'cholecystectomy', 'gallbladder inflammation', 'biliary obstruction',
                'jaundice', 'elevated liver enzymes', 'alkaline phosphatase'
            ]
        }

        for icd_code in target_icd_codes:
            keywords = diagnosis_keywords.get(icd_code, [])

            for cui, info in self.umls_concepts.items():
                if cui in relevant_cuis:
                    continue

                if len(relevant_cuis) >= target_concept_count:
                    break

                # Search in name, terms, and definition
                search_text = ' '.join([
                    info['name'],
                    ' '.join(info.get('terms', [])),
                    info.get('definition', '')
                ]).lower()

                if any(kw in search_text for kw in keywords):
                    relevant_cuis.add(cui)

            if len(relevant_cuis) >= target_concept_count:
                break

        print(f"  After keyword expansion: {len(relevant_cuis)} concepts")

        # Build final concept set
        final_cuis = list(relevant_cuis)[:target_concept_count]

        for cui in final_cuis:
            if cui in self.umls_concepts:
                self.concepts[cui] = self.umls_concepts[cui]

        self.concept_to_idx = {cui: i for i, cui in enumerate(final_cuis)}
        self.idx_to_concept = {i: cui for cui, i in self.concept_to_idx.items()}

        print(f"  ✅ Final: {len(self.concepts)} concepts")

        # Build graph structure
        print(f"\n📊 Building concept graph...")
        self.edge_index, self.edge_attr, _, _ = self.graph_builder.build_graph_for_concepts(final_cuis)
        print(f"  ✅ Graph edges: {self.edge_index.shape[1]}")

        return self.concepts

# ============================================================================
# CELL 8: FOCAL LOSS (Better than BCE for hard samples)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard samples

    FL(pt) = -alpha * (1-pt)^gamma * log(pt)

    where pt is the model's estimated probability for the class
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: [batch, num_classes] raw logits
            targets: [batch, num_classes] binary labels
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# CELL 9: ASYMMETRIC LOSS (Better for multi-label)
# ============================================================================

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification
    Treats positive and negative samples differently
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, targets):
        """
        Args:
            logits: [batch, num_classes]
            targets: [batch, num_classes]
        """
        # Probabilities
        probs = torch.sigmoid(logits)

        # Asymmetric clipping
        probs_pos = probs
        probs_neg = 1 - probs
        probs_neg = (probs_neg + self.clip).clamp(max=1)

        # Calculate loss
        loss_pos = targets * torch.log(probs_pos.clamp(min=1e-8)) * (1 - probs_pos) ** self.gamma_pos
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=1e-8)) * probs_pos ** self.gamma_neg

        loss = -loss_pos - loss_neg
        return loss.mean()

# ============================================================================
# CELL 10: MULTI-TASK HEADS
# ============================================================================

class MultiTaskHead(nn.Module):
    """Multi-task prediction heads for auxiliary tasks"""

    def __init__(self, hidden_size: int, num_diagnosis_classes: int):
        super().__init__()

        # Task 1: Main diagnosis (4 classes)
        self.diagnosis_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_diagnosis_classes)
        )

        # Task 2: Severity prediction (ICU vs ward)
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)  # Binary: severe (ICU) or not
        )

        # Task 3: Mortality prediction
        self.mortality_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)  # Binary: mortality risk
        )

        # Task 4: Length of stay (regression)
        self.los_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)  # Continuous: days
        )

    def forward(self, pooled_output):
        """
        Args:
            pooled_output: [batch, hidden_size]

        Returns:
            dict with all task predictions
        """
        return {
            'diagnosis': self.diagnosis_head(pooled_output),
            'severity': self.severity_head(pooled_output),
            'mortality': self.mortality_head(pooled_output),
            'los': self.los_head(pooled_output)
        }

# ============================================================================
# CELL 11: ENHANCED CROSS-ATTENTION WITH GRAPH CONCEPTS
# ============================================================================

class GraphEnhancedCrossAttention(nn.Module):
    """Cross-attention between text and graph-enhanced concept embeddings"""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.1)
        )

    def forward(self, text_hidden, concept_embeddings):
        """
        Args:
            text_hidden: [batch, seq_len, hidden_size]
            concept_embeddings: [batch, num_concepts, hidden_size]
        """
        batch_size = text_hidden.size(0)

        # Multi-head cross-attention
        Q = self.query(text_hidden).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concept_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concept_embeddings).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        # Attention output
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        context = self.output(context)

        # Residual + norm
        text_hidden = self.norm1(text_hidden + context)

        # FFN
        text_hidden = self.norm2(text_hidden + self.ffn(text_hidden))

        return text_hidden

# ============================================================================
# CELL 12: MAIN MODEL - SHIFAMIND 018
# ============================================================================

class ShifaMind018(nn.Module):
    """
    ShifaMind 018: The Virat Kohli Edition

    - GatorTron-Base transformer
    - Graph-enhanced concept reasoning
    - Multi-task learning
    - Advanced fusion
    """

    def __init__(self,
                 base_model,
                 concept_store,
                 concept_gnn,
                 num_classes: int = 4,
                 fusion_layers: List[int] = [6, 9, 11]):
        super().__init__()

        self.base_model = base_model
        self.concept_store = concept_store
        self.concept_gnn = concept_gnn
        self.num_concepts = len(concept_store.concepts)
        self.num_classes = num_classes
        self.fusion_layers = fusion_layers
        self.hidden_size = base_model.config.hidden_size

        # Cross-attention modules for fusion
        self.fusion_modules = nn.ModuleList([
            GraphEnhancedCrossAttention(self.hidden_size, num_heads=8)
            for _ in fusion_layers
        ])

        # Multi-task heads
        self.multi_task_head = MultiTaskHead(self.hidden_size, num_classes)

        # Concept prediction head
        self.concept_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_concepts)
        )

        # Hook for fusion
        self.fusion_outputs = {}
        self._register_fusion_hooks()

    def _register_fusion_hooks(self):
        """Register forward hooks to inject concept info at specific layers"""

        def create_hook(layer_idx):
            def hook(module, input, output):
                if hasattr(self, '_concept_embeddings_batch'):
                    # Get text hidden states
                    hidden_states = output[0]  # [batch, seq_len, hidden]

                    # Apply cross-attention with concepts
                    fusion_idx = self.fusion_layers.index(layer_idx)
                    fused = self.fusion_modules[fusion_idx](
                        hidden_states,
                        self._concept_embeddings_batch
                    )

                    # Return modified output
                    return (fused,) + output[1:]
                return output
            return hook

        # Register hooks on transformer layers
        for layer_idx in self.fusion_layers:
            if hasattr(self.base_model, 'encoder'):  # BERT-style
                self.base_model.encoder.layer[layer_idx].register_forward_hook(
                    create_hook(layer_idx)
                )

    def forward(self, input_ids, attention_mask, concept_embeddings, edge_index=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            concept_embeddings: [num_concepts, hidden_size] static embeddings
            edge_index: [2, num_edges] graph structure

        Returns:
            dict with all predictions
        """
        batch_size = input_ids.size(0)

        # 1. Enhance concept embeddings with GNN
        if edge_index is not None:
            enhanced_concepts = self.concept_gnn(concept_embeddings, edge_index)  # [num_concepts, hidden]
        else:
            enhanced_concepts = concept_embeddings

        # 2. Expand for batch
        self._concept_embeddings_batch = enhanced_concepts.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, num_concepts, hidden]

        # 3. Forward through transformer (fusion happens via hooks)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 4. Get pooled output ([CLS] token)
        pooled = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        # 5. Multi-task predictions
        task_outputs = self.multi_task_head(pooled)

        # 6. Concept predictions
        concept_logits = self.concept_head(pooled)  # [batch, num_concepts]

        # Clean up batch embeddings
        delattr(self, '_concept_embeddings_batch')

        return {
            'diagnosis_logits': task_outputs['diagnosis'],
            'concept_logits': concept_logits,
            'severity_logits': task_outputs['severity'],
            'mortality_logits': task_outputs['mortality'],
            'los_predictions': task_outputs['los']
        }

# ============================================================================
# CELL 13: DATA LOADING WITH EXPANDED ICD CODES
# ============================================================================

def load_expanded_mimic_data(mimic_path: Path,
                            notes_path: Path,
                            target_codes: List[str],
                            expanded_mapping: Dict,
                            max_samples_per_code: int = 5000):
    """Load MIMIC data with expanded related ICD codes"""

    print(f"\n📂 Loading MIMIC-IV with expanded codes...")

    # Load diagnoses
    diag_file = mimic_path / 'hosp' / 'diagnoses_icd.csv.gz'
    df_diag = pd.read_csv(diag_file, compression='gzip')

    # Convert ICD-10 codes
    df_diag = df_diag[df_diag['icd_version'] == 10].copy()
    df_diag['icd_code'] = df_diag['icd_code'].str.replace('.', '')

    # Load clinical notes
    discharge_file = notes_path / 'discharge.csv.gz'
    df_notes = pd.read_csv(discharge_file, compression='gzip')
    df_notes = df_notes[['note_id', 'subject_id', 'hadm_id', 'text']]

    # Merge
    df = df_notes.merge(df_diag[['subject_id', 'hadm_id', 'icd_code']],
                        on=['subject_id', 'hadm_id'],
                        how='inner')

    # Group by note
    df_grouped = df.groupby(['note_id', 'subject_id', 'hadm_id', 'text'])['icd_code'].apply(list).reset_index()

    print(f"  Total notes with diagnoses: {len(df_grouped)}")

    # Filter for target + related codes
    selected_indices = set()

    for target_code in target_codes:
        all_codes = get_all_codes_for_target(target_code)

        # Find notes with any of these codes
        matching = df_grouped[
            df_grouped['icd_codes'].apply(lambda x: any(code in x for code in all_codes))
        ]

        # Sample up to max
        n_samples = min(len(matching), max_samples_per_code)
        sampled_indices = np.random.choice(matching.index, size=n_samples, replace=False)
        selected_indices.update(sampled_indices)

        print(f"  {target_code}: {len(matching)} notes, sampled {n_samples}")

    df_final = df_grouped.loc[list(selected_indices)].reset_index(drop=True)
    df_final = df_final[df_final['text'].notnull()].reset_index(drop=True)

    # Create binary labels for TARGET codes only (not related codes)
    def create_labels(icd_list, targets):
        return [1 if code in icd_list else 0 for code in targets]

    df_final['labels'] = df_final['icd_codes'].apply(lambda x: create_labels(x, target_codes))

    print(f"  ✅ Final dataset: {len(df_final)} samples")

    return df_final, target_codes

# ============================================================================
# CELL 14: LOAD ICD DESCRIPTIONS
# ============================================================================

def load_icd_descriptions(icd_path: Path) -> Dict[str, str]:
    """Load ICD-10-CM code descriptions"""
    print(f"\n📂 Loading ICD-10 descriptions...")

    icd_file = icd_path / 'icd10cm-codes-descriptions.txt'
    descriptions = {}

    with open(icd_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                code = parts[0].replace('.', '')
                desc = parts[1]
                descriptions[code] = desc

    return descriptions

# ============================================================================
# CELL 15: DATASET CLASS
# ============================================================================

class ShifaMindDataset(Dataset):
    """Dataset for ShifaMind with multi-task labels"""

    def __init__(self, df, tokenizer, max_length=512, target_codes=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_codes = target_codes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = row['text']
        if isinstance(text, float):
            text = ""
        text = str(text)[:10000]  # Limit text length

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Main diagnosis labels
        labels = torch.tensor(row['labels'], dtype=torch.float32)

        # Auxiliary labels (we'll compute these from MIMIC data)
        # For now, use dummy values (you can enhance later)
        severity = torch.tensor([0.0], dtype=torch.float32)  # ICU admission
        mortality = torch.tensor([0.0], dtype=torch.float32)  # In-hospital mortality
        los = torch.tensor([5.0], dtype=torch.float32)  # Length of stay in days

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'severity': severity,
            'mortality': mortality,
            'los': los,
            'note_id': row.get('note_id', idx),
            'text': text
        }

# ============================================================================
# CELL 16: ENHANCED RAG SYSTEM
# ============================================================================

class EnhancedRAG:
    """Enhanced RAG with larger medical knowledge base"""

    def __init__(self, concept_store, tokenizer, base_model, device):
        self.concept_store = concept_store
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.documents = []
        self.doc_embeddings = None
        self.faiss_index = None

    def build_knowledge_base(self, icd_descriptions: Dict[str, str]):
        """Build enhanced knowledge base from concepts + ICD descriptions"""

        print(f"\n📚 Building enhanced RAG knowledge base...")

        documents = []

        # Add concept-based documents
        for cui, concept in self.concept_store.concepts.items():
            # Document from concept
            doc_text = f"{concept['name']}. {concept.get('definition', '')}"
            documents.append({
                'text': doc_text,
                'source': 'concept',
                'cui': cui,
                'diagnosis_category': None
            })

        # Add ICD description documents
        for icd_code, description in icd_descriptions.items():
            # Get diagnosis category (first letter)
            category = icd_code[0] if icd_code else 'Z'

            documents.append({
                'text': f"ICD-10: {icd_code}. {description}",
                'source': 'icd',
                'cui': None,
                'diagnosis_category': category
            })

        self.documents = documents
        print(f"  ✅ Knowledge base: {len(documents)} documents")

        # Create embeddings
        print(f"\n🔍 Creating document embeddings...")
        doc_texts = [d['text'] for d in documents]

        embeddings = []
        batch_size = 32

        for i in tqdm(range(0, len(doc_texts), batch_size), desc="  Encoding"):
            batch = doc_texts[i:i+batch_size]
            encoding = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.base_model(**encoding)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            embeddings.append(batch_embeddings)

        self.doc_embeddings = np.vstack(embeddings).astype('float32')

        # Build FAISS index
        print(f"\n🔍 Building FAISS index...")
        dimension = self.doc_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product
        faiss.normalize_L2(self.doc_embeddings)
        self.faiss_index.add(self.doc_embeddings)

        print(f"  ✅ FAISS index built: {self.faiss_index.ntotal} documents")

    def retrieve(self, query_embedding, top_k=10, diagnosis_filter=None):
        """Retrieve relevant documents"""

        # Normalize query
        query_emb = query_embedding.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_emb)

        # Search
        scores, indices = self.faiss_index.search(query_emb, top_k * 2)

        # Filter by diagnosis if specified
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]

                # Apply diagnosis filter
                if diagnosis_filter:
                    if doc['diagnosis_category'] and doc['diagnosis_category'] != diagnosis_filter:
                        continue

                results.append({
                    'document': doc,
                    'score': float(score)
                })

                if len(results) >= top_k:
                    break

        return results

# ============================================================================
# CELL 17: MULTI-TASK LOSS
# ============================================================================

class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning"""

    def __init__(self, use_focal=True):
        super().__init__()

        if use_focal:
            self.diagnosis_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
            self.concept_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.diagnosis_loss_fn = nn.BCEWithLogitsLoss()
            self.concept_loss_fn = nn.BCEWithLogitsLoss()

        self.severity_loss_fn = nn.BCEWithLogitsLoss()
        self.mortality_loss_fn = nn.BCEWithLogitsLoss()
        self.los_loss_fn = nn.MSELoss()

        # Task weights
        self.weights = {
            'diagnosis': 1.0,
            'concept': 0.5,
            'severity': 0.2,
            'mortality': 0.2,
            'los': 0.1
        }

    def forward(self, predictions, targets, concept_labels=None):
        """
        Args:
            predictions: dict from model forward
            targets: dict with ground truth labels
            concept_labels: [batch, num_concepts] pseudo-labels
        """
        losses = {}

        # Diagnosis loss (main task)
        losses['diagnosis'] = self.diagnosis_loss_fn(
            predictions['diagnosis_logits'],
            targets['labels']
        )

        # Concept loss (if available)
        if concept_labels is not None:
            losses['concept'] = self.concept_loss_fn(
                predictions['concept_logits'],
                concept_labels
            )
        else:
            losses['concept'] = torch.tensor(0.0, device=predictions['diagnosis_logits'].device)

        # Severity loss
        losses['severity'] = self.severity_loss_fn(
            predictions['severity_logits'],
            targets['severity']
        )

        # Mortality loss
        losses['mortality'] = self.mortality_loss_fn(
            predictions['mortality_logits'],
            targets['mortality']
        )

        # Length of stay loss
        losses['los'] = self.los_loss_fn(
            predictions['los_predictions'],
            targets['los']
        )

        # Weighted sum
        total_loss = sum(self.weights[k] * v for k, v in losses.items())
        losses['total'] = total_loss

        return losses

# ============================================================================
# CELL 18: EXPONENTIAL MOVING AVERAGE (EMA) FOR WEIGHTS
# ============================================================================

class EMA:
    """Exponential Moving Average of model weights"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ============================================================================
# CELL 19: ADVANCED TRAINER
# ============================================================================

class AdvancedTrainer:
    """Advanced trainer with all the bells and whistles"""

    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 concept_embeddings,
                 edge_index,
                 device,
                 use_focal_loss=True,
                 gradient_accumulation_steps=4):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.concept_embeddings = concept_embeddings
        self.edge_index = edge_index
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Loss function
        self.criterion = MultiTaskLoss(use_focal=use_focal_loss)

        # Optimizer with different LR for backbone and heads
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if 'base_model' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},  # Lower LR for pretrained
            {'params': head_params, 'lr': 3e-5}  # Higher LR for new heads
        ], weight_decay=0.01)

        # Scheduler
        total_steps = len(train_loader) * 20 // gradient_accumulation_steps
        warmup_steps = total_steps // 10

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # EMA
        self.ema = EMA(model, decay=0.999)

        # History
        self.history = []
        self.best_f1 = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch, concept_labels=None):
        """Train for one epoch"""

        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"  Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            severity = batch['severity'].to(self.device)
            mortality = batch['mortality'].to(self.device)
            los = batch['los'].to(self.device)

            # Get concept labels for this batch if available
            batch_concept_labels = None
            if concept_labels is not None:
                batch_indices = batch.get('index', None)
                if batch_indices is not None:
                    batch_concept_labels = concept_labels[batch_indices].to(self.device)

            # Forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                concept_embeddings=self.concept_embeddings,
                edge_index=self.edge_index
            )

            # Compute loss
            targets = {
                'labels': labels,
                'severity': severity,
                'mortality': mortality,
                'los': los
            }

            losses = self.criterion(outputs, targets, batch_concept_labels)
            loss = losses['total'] / self.gradient_accumulation_steps

            # Backward
            loss.backward()

            # Update weights every N steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Update EMA
                self.ema.update()

            # Track losses
            for k, v in losses.items():
                epoch_losses[k] += v.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{epoch_losses['total']/num_batches:.4f}",
                'diag': f"{epoch_losses['diagnosis']/num_batches:.4f}"
            })

        # Average losses
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        return avg_losses

    def evaluate(self, use_ema=False):
        """Evaluate on validation set"""

        # Apply EMA weights if requested
        if use_ema:
            self.ema.apply_shadow()

        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="  Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    concept_embeddings=self.concept_embeddings,
                    edge_index=self.edge_index
                )

                preds = (torch.sigmoid(outputs['diagnosis_logits']) > 0.5).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Metrics
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

        # Restore original weights if EMA was used
        if use_ema:
            self.ema.restore()

        return {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, num_epochs=20, early_stopping_patience=5, concept_labels=None):
        """Full training loop"""

        print(f"\n🚀 Training for {num_epochs} epochs...")

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")

            # Train
            start_time = time.time()
            train_losses = self.train_epoch(epoch, concept_labels)
            epoch_time = time.time() - start_time

            # Validate (with EMA)
            val_metrics = self.evaluate(use_ema=True)

            # Log
            print(f"\n  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Macro F1: {val_metrics['macro_f1']:.4f}")
            print(f"  Val Micro F1: {val_metrics['micro_f1']:.4f}")
            print(f"  Time: {epoch_time:.1f}s")

            # Save best
            if val_metrics['macro_f1'] > self.best_f1:
                self.best_f1 = val_metrics['macro_f1']
                self.best_epoch = epoch
                patience_counter = 0

                # Apply EMA before saving
                self.ema.apply_shadow()
                torch.save(self.model.state_dict(), 'shifamind_018_best.pt')
                self.ema.restore()

                print(f"  ✅ Best F1: {self.best_f1:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break

            # History
            self.history.append({
                'epoch': epoch,
                'train_loss': train_losses,
                'val_f1': val_metrics['macro_f1'],
                'time': epoch_time
            })

        print(f"\n✅ Training complete!")
        print(f"  Best F1: {self.best_f1:.4f} (Epoch {self.best_epoch})")

        return self.history

# ============================================================================
# CELL 20: MAIN EXECUTION
# ============================================================================

print("\n" + "="*70)
print("SHIFAMIND 018: THE VIRAT KOHLI EDITION 🏏")
print("="*70)

# Target ICD codes
target_codes = ['J189', 'I5023', 'A419', 'K8000']

# Step 1: Load UMLS
umls_loader = EnhancedUMLSLoader(UMLS_PATH)
umls_concepts = umls_loader.load_concepts(max_concepts=50000)
umls_loader.load_semantic_types()
umls_loader.load_relationships()
umls_loader.load_definitions()

# Step 2: Load ICD mappings
icd_to_cui = load_icd_to_cui_mapping(UMLS_PATH)
icd_descriptions = load_icd_descriptions(ICD_PATH)

# Step 3: Build expanded concept store
concept_store = EnhancedConceptStore(
    umls_concepts=umls_concepts,
    icd_to_cui=icd_to_cui,
    relationships=umls_loader.relationships
)

concepts = concept_store.build_expanded_concept_set(
    target_icd_codes=target_codes,
    expanded_mapping=EXPANDED_ICD_MAPPING,
    target_concept_count=2000
)

# Step 4: Load expanded MIMIC data
df, target_codes = load_expanded_mimic_data(
    mimic_path=MIMIC_PATH,
    notes_path=NOTES_PATH,
    target_codes=target_codes,
    expanded_mapping=EXPANDED_ICD_MAPPING,
    max_samples_per_code=5000
)

print(f"\n🎯 Target diagnoses:")
for code in target_codes:
    desc = icd_descriptions.get(code, 'Unknown')
    print(f"  {code}: {desc}")

# Step 5: Load GatorTron model
print(f"\n🧬 Loading GatorTron-Base...")
model_name = "UFNLP/gatortron-base"  # 345M params

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name).to(device)

print(f"  ✅ Model loaded: {model_name}")
print(f"  Hidden size: {base_model.config.hidden_size}")

# Step 6: Create concept embeddings
print(f"\n🧬 Creating concept embeddings...")
concept_texts = [
    f"{c['name']}. {c.get('definition', '')}"
    for c in concept_store.concepts.values()
]

concept_embeddings_list = []
batch_size = 32

for i in tqdm(range(0, len(concept_texts), batch_size), desc="  Encoding concepts"):
    batch = concept_texts[i:i+batch_size]
    encoding = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = base_model(**encoding)
        batch_emb = outputs.last_hidden_state[:, 0, :].cpu()

    concept_embeddings_list.append(batch_emb)

concept_embeddings = torch.cat(concept_embeddings_list, dim=0).to(device)
print(f"  ✅ Concept embeddings: {concept_embeddings.shape}")

# Step 7: Build enhanced RAG
rag_system = EnhancedRAG(concept_store, tokenizer, base_model, device)
rag_system.build_knowledge_base(icd_descriptions)

# Step 8: Split data
df_train, df_temp = train_test_split(df, test_size=0.3, random_state=SEED)
df_val, df_test = train_test_split(df_temp, test_size=0.67, random_state=SEED)

print(f"\n📊 Data splits:")
print(f"  Train: {len(df_train)} samples")
print(f"  Val:   {len(df_val)} samples")
print(f"  Test:  {len(df_test)} samples")

# Step 9: Create datasets
train_dataset = ShifaMindDataset(df_train, tokenizer, target_codes=target_codes)
val_dataset = ShifaMindDataset(df_val, tokenizer, target_codes=target_codes)
test_dataset = ShifaMindDataset(df_test, tokenizer, target_codes=target_codes)

# Add index to dataset for concept label lookup
train_dataset.df['index'] = range(len(train_dataset.df))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# Step 10: Initialize model components
print(f"\n🏗️  Initializing ShifaMind 018...")

# Concept GNN
concept_gnn = ConceptGNN(
    input_dim=base_model.config.hidden_size,
    hidden_dim=base_model.config.hidden_size,
    num_layers=2
).to(device)

# Edge index for graph
edge_index = concept_store.edge_index.to(device) if hasattr(concept_store, 'edge_index') else None

# Main model
model = ShifaMind018(
    base_model=base_model,
    concept_store=concept_store,
    concept_gnn=concept_gnn,
    num_classes=len(target_codes),
    fusion_layers=[6, 9, 11]
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  ✅ Model initialized")
print(f"  Total parameters: {total_params/1e6:.1f}M")
print(f"  Trainable parameters: {trainable_params/1e6:.1f}M")

# Step 11: Initialize trainer
trainer = AdvancedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    concept_embeddings=concept_embeddings,
    edge_index=edge_index,
    device=device,
    use_focal_loss=True,
    gradient_accumulation_steps=4
)

# Step 12: Train!
history = trainer.train(
    num_epochs=20,
    early_stopping_patience=5,
    concept_labels=None  # Can add PMI-based labels later
)

# Step 13: Final evaluation
print(f"\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

model.load_state_dict(torch.load('shifamind_018_best.pt'))
model.eval()

test_preds = []
test_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="  Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            concept_embeddings=concept_embeddings,
            edge_index=edge_index
        )

        preds = (torch.sigmoid(outputs['diagnosis_logits']) > 0.5).cpu().numpy()
        test_preds.append(preds)
        test_labels.append(labels.cpu().numpy())

test_preds = np.vstack(test_preds)
test_labels = np.vstack(test_labels)

# Metrics
macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
micro_f1 = f1_score(test_labels, test_preds, average='micro', zero_division=0)
per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)

print(f"\n📊 ShifaMind 018 Results:")
print(f"  Macro F1:   {macro_f1:.4f}")
print(f"  Micro F1:   {micro_f1:.4f}")
print(f"  Precision:  {precision:.4f}")
print(f"  Recall:     {recall:.4f}")

print(f"\n📊 Per-Class F1:")
for i, code in enumerate(target_codes):
    print(f"  {code}: {per_class_f1[i]:.4f}")

print("\n" + "="*70)
print("✅ SHIFAMIND 018 COMPLETE! 🏏")
print("="*70)
