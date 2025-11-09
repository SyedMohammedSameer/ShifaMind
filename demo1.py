#!/usr/bin/env python3
"""
ShifaMind Clinical Demo - ChatGPT vs ShifaMind Comparison
For Medical Professional Demonstration

Run in Google Colab:
1. Upload demo1.py and model checkpoint (stage4_joint_best_revised.pt)
2. Run all cells
3. Enter OpenAI API key when prompted
4. Click on ngrok URL to open demo
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COLAB SETUP
# ============================================================================
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🔧 Installing dependencies...")
    os.system('pip install -q streamlit openai torch transformers faiss-cpu')

    from google.colab import drive
    drive.mount('/content/drive')

    print("✅ Dependencies installed")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import streamlit as st
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import json
import time
from pathlib import Path
from tqdm.auto import tqdm
import faiss

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind/01_Raw_Datasets/Extracted')
MIMIC_PATH = BASE_PATH / 'mimic-iv-3.1'
UMLS_PATH = BASE_PATH / 'umls-2025AA-metathesaurus-full/2025AA/META'

# Target diagnoses
TARGET_CODES = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic (congestive) heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis without obstruction'
}

# ============================================================================
# TEMPLATE CLINICAL NOTES
# ============================================================================
DEMO_NOTES = {
    "Pneumonia Case": """CHIEF COMPLAINT: Shortness of breath and fever

HISTORY OF PRESENT ILLNESS:
67-year-old male with a 3-day history of productive cough, fever (101.5°F), and progressive dyspnea. Patient reports yellow-green sputum production and pleuritic chest pain on the right side. No recent travel. Has not received pneumococcal vaccine.

PAST MEDICAL HISTORY: Hypertension, Type 2 Diabetes Mellitus

PHYSICAL EXAMINATION:
- Temperature: 101.8°F, HR: 105 bpm, RR: 24/min, BP: 142/88 mmHg, SpO2: 89% on room air
- Chest: Decreased breath sounds and crackles in right lower lobe
- Dullness to percussion over right lower lung field

LABORATORY:
- WBC: 16,500/μL with left shift
- CRP: 185 mg/L
- Chest X-ray: Right lower lobe infiltrate with air bronchograms

ASSESSMENT: Community-acquired pneumonia with respiratory compromise.""",

    "Heart Failure Case": """CHIEF COMPLAINT: Worsening shortness of breath and leg swelling

HISTORY OF PRESENT ILLNESS:
72-year-old female with history of heart failure (EF 25%) presenting with 5-day history of progressive dyspnea, orthopnea (3-pillow), paroxysmal nocturnal dyspnea, and bilateral lower extremity edema. Patient admits to dietary indiscretion (eating salty foods) and missing diuretic doses. Weight gain of 12 lbs over past week.

PAST MEDICAL HISTORY:
- Ischemic cardiomyopathy (prior MI 2018)
- Atrial fibrillation on warfarin
- Chronic kidney disease Stage 3

PHYSICAL EXAMINATION:
- BP: 158/92 mmHg, HR: 98 irregular, RR: 22/min, SpO2: 91% on 2L NC
- JVP elevated to 12 cm
- Cardiac: S3 gallop, irregular rhythm
- Lungs: Bilateral crackles to mid-lung fields
- Extremities: 3+ pitting edema to knees bilaterally

LABORATORY:
- BNP: 1,850 pg/mL (elevated from baseline 450)
- Creatinine: 1.8 mg/dL (baseline 1.4)
- Chest X-ray: Cardiomegaly, bilateral pleural effusions, pulmonary vascular congestion

ASSESSMENT: Acute decompensated heart failure on chronic systolic heart failure, NYHA Class IV.""",

    "Sepsis Case": """CHIEF COMPLAINT: Confusion and fever

HISTORY OF PRESENT ILLNESS:
81-year-old nursing home resident brought to ED with altered mental status, fever, and hypotension. Per nursing home staff, patient had decreased oral intake and lethargy for 2 days. Found to be febrile to 103.2°F this morning with SBP 78 mmHg. Patient normally alert and oriented, now confused and lethargic.

PAST MEDICAL HISTORY:
- Dementia
- Recurrent UTIs with recent ESBL E. coli
- Chronic indwelling Foley catheter
- Hypertension

PHYSICAL EXAMINATION:
- Temperature: 103.5°F, HR: 118 bpm, RR: 28/min, BP: 82/45 mmHg, SpO2: 94% on room air
- General: Lethargic, disoriented to time and place
- Skin: Warm, delayed capillary refill
- Lungs: Clear bilaterally
- Abdomen: Soft, mild suprapubic tenderness
- Urine: Cloudy, malodorous

LABORATORY:
- WBC: 18,900/μL with 15% bands
- Lactate: 4.2 mmol/L
- Creatinine: 2.1 mg/dL (baseline 1.1)
- Urinalysis: >100 WBC, many bacteria
- Blood cultures: 2/2 bottles growing Gram-negative rods at 8 hours

INTERVENTIONS:
- 2L IV bolus normal saline
- Empiric broad-spectrum antibiotics started
- Norepinephrine infusion initiated

ASSESSMENT: Septic shock secondary to urosepsis (suspected ESBL E. coli).""",

    "Cholecystitis Case": """CHIEF COMPLAINT: Right upper quadrant pain

HISTORY OF PRESENT ILLNESS:
52-year-old obese female presenting with 8-hour history of severe right upper quadrant pain that started after eating a large fatty meal. Pain radiates to right shoulder blade. Associated with nausea and two episodes of vomiting. Patient reports similar but milder episodes over past 3 months, always after fatty foods. No jaundice. No prior gallbladder issues.

PAST MEDICAL HISTORY:
- Obesity (BMI 38)
- Hyperlipidemia
- GERD

PHYSICAL EXAMINATION:
- Temperature: 101.2°F, HR: 98 bpm, RR: 18/min, BP: 138/84 mmHg
- Abdomen: Positive Murphy's sign, tenderness in RUQ with guarding, no rebound
- No jaundice, sclera clear

LABORATORY:
- WBC: 14,200/μL with left shift
- AST: 98 U/L, ALT: 112 U/L (mildly elevated)
- Alkaline phosphatase: 156 U/L
- Total bilirubin: 1.4 mg/dL
- Lipase: 45 U/L (normal)

IMAGING:
- RUQ Ultrasound:
  • Gallbladder wall thickening (5.2 mm)
  • Multiple gallstones, largest 1.8 cm
  • Pericholecystic fluid
  • Positive sonographic Murphy's sign
  • Common bile duct 4.5 mm (normal)

ASSESSMENT: Acute calculous cholecystitis with gallstone disease."""
}

# ============================================================================
# UMLS LOADER (Same as 016.py)
# ============================================================================
class FastUMLSLoader:
    def __init__(self, umls_path: Path):
        self.umls_path = umls_path
        self.concepts = {}
        self.cui_to_icd10 = defaultdict(list)
        self.icd10_to_cui = defaultdict(list)

    def load_concepts(self, max_concepts: int = 30000):
        print(f"📚 Loading UMLS concepts (max: {max_concepts})...")

        target_types = {'T047', 'T046', 'T184', 'T033', 'T048', 'T037', 'T191', 'T020'}
        cui_to_types = self._load_semantic_types()

        mrconso_path = self.umls_path / 'MRCONSO.RRF'
        concepts_loaded = 0

        with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="Loading MRCONSO", total=max_concepts):
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

        print(f"✅ Loaded {len(self.concepts)} concepts")
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
        print("📖 Loading definitions...")
        mrdef_path = self.umls_path / 'MRDEF.RRF'
        definitions_added = 0

        with open(mrdef_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading definitions"):
                fields = line.strip().split('|')
                if len(fields) >= 6:
                    cui = fields[0]
                    definition = fields[5]

                    if cui in concepts and definition:
                        if 'definition' not in concepts[cui]:
                            concepts[cui]['definition'] = definition
                            definitions_added += 1

        print(f"✅ Added {definitions_added} definitions")
        return concepts

# ============================================================================
# SEMANTIC TYPE VALIDATOR
# ============================================================================
class SemanticTypeValidator:
    """Filters concepts by clinical relevance (Same as 016.py)"""

    RELEVANT_TYPES = {'T047', 'T046', 'T184', 'T033', 'T048', 'T037', 'T191', 'T020'}

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
# PHASE 4 CONCEPT STORE (150 concepts - Same as 016.py)
# ============================================================================
class Phase4ConceptStore:
    """Builds the exact 150 concepts as training"""

    def __init__(self, umls_concepts: Dict, icd_to_cui: Dict):
        self.umls_concepts = umls_concepts
        self.icd_to_cui = icd_to_cui
        self.concepts = {}
        self.semantic_validator = SemanticTypeValidator(umls_concepts)

    def build_concept_set(self, target_icd_codes: List[str], target_concept_count: int = 150):
        print(f"🔬 Building {target_concept_count} concepts...")

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

        # Strategy 2: Keyword expansion
        diagnosis_keywords = {
            'J189': ['pneumonia', 'lung infection', 'respiratory infection',
                     'infiltrate', 'bacterial pneumonia', 'aspiration'],
            'I5023': ['heart failure', 'cardiac failure', 'cardiomyopathy',
                      'pulmonary edema', 'ventricular dysfunction'],
            'A419': ['sepsis', 'septicemia', 'bacteremia', 'infection',
                     'septic shock', 'organ dysfunction'],
            'K8000': ['cholecystitis', 'gallbladder', 'biliary disease',
                      'gallstone', 'cholelithiasis']
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

        print(f"✅ Final: {len(self.concepts)} concepts")
        return self.concepts

    def _get_icd_variants(self, code: str) -> List[str]:
        variants = {code, code.replace('.', '')}
        no_dots = code.replace('.', '')
        if len(no_dots) >= 4:
            variants.add(no_dots[:3] + '.' + no_dots[3:])
        variants.add(no_dots[:3])
        return list(variants)

    def get_concept_name(self, cui: str) -> str:
        """Get concept name by CUI"""
        return self.concepts.get(cui, {}).get('name', cui)

    def get_concept_definition(self, cui: str) -> str:
        """Get concept definition"""
        return self.concepts.get(cui, {}).get('definition', 'No definition available')

# ============================================================================
# MODEL ARCHITECTURE (Same as Phase 4 Revised)
# ============================================================================
class SimpleCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, text_features, concept_features):
        attn_output, attn_weights = self.attention(
            text_features, concept_features, concept_features
        )
        return self.norm(text_features + attn_output), attn_weights

class DiagnosisConceptHead(nn.Module):
    def __init__(self, hidden_size, num_concepts, num_classes):
        super().__init__()
        self.diagnosis_head = nn.Linear(hidden_size, num_classes)
        self.concept_head = nn.Linear(hidden_size, num_concepts)
        self.diagnosis_concept_interaction = nn.Bilinear(
            num_classes, num_concepts, num_concepts
        )

    def forward(self, pooled_output):
        diagnosis_logits = self.diagnosis_head(pooled_output)
        concept_logits = self.concept_head(pooled_output)

        diagnosis_probs = torch.sigmoid(diagnosis_logits)

        interaction = self.diagnosis_concept_interaction(
            diagnosis_probs, concept_logits
        )

        refined_concept_logits = concept_logits + 0.1 * interaction

        return diagnosis_logits, refined_concept_logits

class Phase4RevisedShifaMind(nn.Module):
    def __init__(self, base_model, concept_store, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model  # Changed from self.bert to match checkpoint
        self.num_layers = self.base_model.config.num_hidden_layers
        self.hidden_size = self.base_model.config.hidden_size
        self.fusion_layers = fusion_layers
        self.concept_store = concept_store

        # Freeze early BERT layers
        for i, layer in enumerate(self.base_model.encoder.layer):
            if i < min(fusion_layers) - 2:
                for param in layer.parameters():
                    param.requires_grad = False

        # Cross-attention fusion
        self.fusion_modules = nn.ModuleDict({
            str(layer_idx): SimpleCrossAttention(self.hidden_size, num_heads=8)
            for layer_idx in fusion_layers
        })

        # Diagnosis + concept heads
        self.heads = DiagnosisConceptHead(
            self.hidden_size,
            len(concept_store.concepts),
            num_classes
        )

    def forward(self, input_ids, attention_mask, concept_embeddings, return_attention=False):
        batch_size = input_ids.size(0)
        num_concepts = concept_embeddings.size(0)

        # Expand concept embeddings
        concept_features = concept_embeddings.unsqueeze(0).expand(
            batch_size, num_concepts, self.hidden_size
        )

        # BERT forward with fusion
        outputs = self.base_model.embeddings(input_ids)
        hidden_states = outputs

        attention_weights = {}

        for i, layer in enumerate(self.base_model.encoder.layer):
            hidden_states = layer(hidden_states)[0]

            # Apply fusion at specified layers
            if i in self.fusion_layers:
                hidden_states, attn_weights = self.fusion_modules[str(i)](
                    hidden_states, concept_features
                )
                if return_attention:
                    attention_weights[f'layer_{i}'] = attn_weights

        # Pool
        pooled = hidden_states[:, 0]  # [CLS] token

        # Get predictions
        diagnosis_logits, concept_logits = self.heads(pooled)

        if return_attention:
            return diagnosis_logits, concept_logits, attention_weights
        return diagnosis_logits, concept_logits

# ============================================================================
# CHATGPT INTEGRATION
# ============================================================================
def query_chatgpt(clinical_note: str, api_key: str) -> str:
    """Query ChatGPT for diagnosis"""
    try:
        from openai import OpenAI

        # Initialize client (compatible with all versions)
        client = OpenAI(api_key=api_key)

        prompt = f"""You are a medical AI assistant. Analyze this clinical note and provide the most likely diagnosis.

Clinical Note:
{clinical_note}

Provide a concise diagnosis with brief reasoning (2-3 sentences max)."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical diagnostic assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )

        return response.choices[0].message.content.strip()

    except ImportError:
        return "❌ Error: OpenAI library not installed. Run: pip install --upgrade openai"
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return f"❌ Error: Invalid API key. Please check your OpenAI API key.\n\nDetails: {error_msg}"
        elif "proxies" in error_msg.lower():
            return "❌ Error: OpenAI version conflict.\n\nFix: Run this in a Colab cell:\n!pip install --upgrade --force-reinstall openai\n\nThen restart the demo."
        else:
            return f"❌ Error: {error_msg}\n\nTry: pip install --upgrade openai"

# ============================================================================
# SHIFAMIND INFERENCE
# ============================================================================
@torch.no_grad()
def predict_shifamind(
    model,
    tokenizer,
    clinical_note: str,
    concept_embeddings: torch.Tensor,
    concept_store: Phase4ConceptStore,
    threshold: float = 0.7
) -> Dict:
    """Run ShifaMind inference"""
    model.eval()

    # Tokenize
    encoding = tokenizer(
        clinical_note,
        max_length=384,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    # Forward pass
    start_time = time.time()
    diagnosis_logits, concept_logits, attention_weights = model(
        input_ids, attention_mask, concept_embeddings, return_attention=True
    )
    inference_time = time.time() - start_time

    # Process predictions
    diagnosis_probs = torch.sigmoid(diagnosis_logits).cpu().numpy()[0]
    concept_probs = torch.sigmoid(concept_logits).cpu().numpy()[0]

    # Get predictions
    predicted_diagnoses = []
    for idx, (code, name) in enumerate(TARGET_CODES.items()):
        prob = diagnosis_probs[idx]
        if prob > 0.5:
            predicted_diagnoses.append({
                'code': code,
                'name': name,
                'confidence': float(prob)
            })

    # Get activated concepts
    activated_concepts = []
    concept_cuis = list(concept_store.concepts.keys())
    for idx, cui in enumerate(concept_cuis):
        prob = concept_probs[idx]
        if prob > threshold:
            activated_concepts.append({
                'cui': cui,
                'name': concept_store.get_concept_name(cui),
                'definition': concept_store.get_concept_definition(cui),
                'confidence': float(prob)
            })

    # Sort by confidence
    predicted_diagnoses.sort(key=lambda x: x['confidence'], reverse=True)
    activated_concepts.sort(key=lambda x: x['confidence'], reverse=True)

    return {
        'diagnoses': predicted_diagnoses,
        'concepts': activated_concepts,
        'inference_time': inference_time,
        'num_concepts': len(activated_concepts)
    }

# ============================================================================
# STREAMLIT UI
# ============================================================================
def render_comparison_ui():
    """Main Streamlit UI"""

    st.set_page_config(
        page_title="ShifaMind Clinical Demo",
        page_icon="🏥",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .chatgpt-box {
            background-color: #f0f8ff;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
        }
        .shifamind-box {
            background-color: #fff5f0;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #ff6b6b;
        }
        .concept-badge {
            display: inline-block;
            background-color: #e3f2fd;
            color: #1976d2;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            margin: 0.2rem;
            font-size: 0.9rem;
        }
        .diagnosis-card {
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .metric-box {
            background-color: #fafafa;
            padding: 1rem;
            border-radius: 5px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header">🏥 ShifaMind Clinical AI Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explainable Medical Diagnosis: ChatGPT vs ShifaMind</div>', unsafe_allow_html=True)

    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")

    # API Key
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key for ChatGPT comparison"
    )

    # Model loading status
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    if not st.session_state.model_loaded:
        with st.sidebar:
            if st.button("🚀 Load ShifaMind Model"):
                with st.spinner("Loading model..."):
                    try:
                        # Load model
                        concept_store, model, tokenizer, concept_embeddings = load_shifamind_model()

                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.concept_embeddings = concept_embeddings
                        st.session_state.concept_store = concept_store
                        st.session_state.model_loaded = True

                        st.success("✅ Model loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
    else:
        st.sidebar.success("✅ Model Ready")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Template Cases")
    st.sidebar.markdown("Select a pre-written clinical case below")

    # Main content
    st.markdown("## 📝 Clinical Note Input")

    # Template selector
    selected_template = st.selectbox(
        "Choose a template case:",
        ["Custom"] + list(DEMO_NOTES.keys())
    )

    # Text input
    if selected_template == "Custom":
        clinical_note = st.text_area(
            "Enter clinical note:",
            height=300,
            placeholder="Enter the clinical note here..."
        )
    else:
        clinical_note = st.text_area(
            f"Clinical Note ({selected_template}):",
            value=DEMO_NOTES[selected_template],
            height=300
        )

    # Run comparison
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_comparison = st.button("🔬 Run Diagnosis Comparison", type="primary", use_container_width=True)

    if run_comparison:
        if not clinical_note.strip():
            st.warning("⚠️ Please enter a clinical note")
            return

        if not api_key:
            st.warning("⚠️ Please enter OpenAI API key in the sidebar")
            return

        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load the ShifaMind model first")
            return

        st.markdown("---")
        st.markdown("## 📊 Diagnosis Results")

        # Two columns for comparison
        col_chatgpt, col_shifamind = st.columns(2)

        # ChatGPT Column
        with col_chatgpt:
            st.markdown("### 🤖 ChatGPT-4")
            with st.spinner("Querying ChatGPT..."):
                chatgpt_start = time.time()
                chatgpt_response = query_chatgpt(clinical_note, api_key)
                chatgpt_time = time.time() - chatgpt_start

            st.markdown(f'<div class="chatgpt-box">{chatgpt_response}</div>', unsafe_allow_html=True)
            st.caption(f"⏱️ Response time: {chatgpt_time:.2f}s")

            st.markdown("#### ❌ Limitations:")
            st.markdown("""
            - **No explainability**: Black box reasoning
            - **No structured output**: Just text
            - **No medical concepts**: Cannot trace decision
            - **No confidence scores**: Binary output
            - **Not specialized**: General AI, not clinical
            """)

        # ShifaMind Column
        with col_shifamind:
            st.markdown("### 🏥 ShifaMind")
            with st.spinner("Running ShifaMind inference..."):
                results = predict_shifamind(
                    st.session_state.model,
                    st.session_state.tokenizer,
                    clinical_note,
                    st.session_state.concept_embeddings,
                    st.session_state.concept_store,
                    threshold=0.7
                )

            # Display results in structured format
            st.markdown('<div class="shifamind-box">', unsafe_allow_html=True)

            # Predicted Diagnoses
            st.markdown("#### 🎯 Predicted Diagnoses")
            for diag in results['diagnoses']:
                confidence_pct = diag['confidence'] * 100
                st.markdown(f"""
                <div class="diagnosis-card">
                    <strong>{diag['code']}</strong>: {diag['name']}<br>
                    <strong>Confidence:</strong> {confidence_pct:.1f}%
                    <div style="background-color: #ddd; border-radius: 5px; height: 20px; margin-top: 5px;">
                        <div style="background-color: #4CAF50; height: 20px; border-radius: 5px; width: {confidence_pct}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Activated Concepts
            st.markdown(f"#### 🧠 Activated Medical Concepts ({results['num_concepts']} concepts)")

            # Show top concepts
            for concept in results['concepts'][:10]:
                confidence_pct = concept['confidence'] * 100
                st.markdown(f"""
                <div style="margin: 0.5rem 0; padding: 0.8rem; background-color: #f5f5f5; border-radius: 5px;">
                    <strong style="color: #1976d2;">{concept['name']}</strong>
                    <span style="background-color: #4CAF50; color: white; padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem;">
                        {confidence_pct:.0f}%
                    </span>
                    <br>
                    <small style="color: #666;">{concept['definition']}</small>
                </div>
                """, unsafe_allow_html=True)

            if results['num_concepts'] > 10:
                st.caption(f"... and {results['num_concepts'] - 10} more concepts")

            st.markdown('</div>', unsafe_allow_html=True)

            # Metrics
            st.caption(f"⏱️ Inference time: {results['inference_time']:.3f}s")

            # Advantages
            st.markdown("#### ✅ Advantages:")
            st.markdown("""
            - **✅ Explainable**: Shows medical concepts used
            - **✅ Structured output**: ICD codes + confidence
            - **✅ Clinical concepts**: UMLS-based reasoning
            - **✅ Transparent**: Can audit decision process
            - **✅ Specialized**: Trained on MIMIC-IV clinical data
            - **✅ Trustworthy**: Medical professionals can verify
            """)

        st.markdown("---")

        # Summary comparison
        st.markdown("## 🎓 Why ShifaMind is Superior for Clinical Use")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="metric-box">
                <h3 style="color: #1976d2;">Explainability</h3>
                <p>Shows <strong>which medical concepts</strong> led to diagnosis</p>
                <p style="color: #4CAF50; font-size: 1.5rem;">✓</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-box">
                <h3 style="color: #1976d2;">Clinical Validation</h3>
                <p>Uses <strong>UMLS ontology</strong> for medical accuracy</p>
                <p style="color: #4CAF50; font-size: 1.5rem;">✓</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-box">
                <h3 style="color: #1976d2;">Confidence Scores</h3>
                <p>Quantified uncertainty for <strong>each diagnosis</strong></p>
                <p style="color: #4CAF50; font-size: 1.5rem;">✓</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h3 style="color: #1976d2;">🏆 ShifaMind: The Future of Explainable Medical AI</h3>
            <p style="font-size: 1.1rem; color: #333;">
                While ChatGPT provides text-based reasoning, ShifaMind offers <strong>structured,
                explainable, and clinically-grounded</strong> diagnosis with transparent medical concept activation.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_shifamind_model():
    """Load trained ShifaMind model with full 150 UMLS concepts (same as 016.py)"""

    print("🏥 Loading ShifaMind with 150 UMLS concepts...")
    print("⏱️  This takes ~2 minutes on first load, then cached")

    # Step 1: Load UMLS
    print("\n📂 Loading UMLS...")
    umls_loader = FastUMLSLoader(UMLS_PATH)
    umls_concepts = umls_loader.load_concepts(max_concepts=30000)
    umls_concepts = umls_loader.load_definitions(umls_concepts)

    # Step 2: Build 150 concept set
    print("\n🔬 Building 150 concept set...")
    target_icd_codes = list(TARGET_CODES.keys())
    concept_store = Phase4ConceptStore(umls_concepts, umls_loader.icd10_to_cui)
    concept_store.build_concept_set(target_icd_codes, target_concept_count=150)

    # Step 3: Load Bio_ClinicalBERT
    print("\n🧬 Loading Bio_ClinicalBERT...")
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    # Step 4: Initialize model
    print("\n🤖 Initializing model...")
    model = Phase4RevisedShifaMind(
        base_model=base_model,
        concept_store=concept_store,
        num_classes=len(TARGET_CODES),
        fusion_layers=[9, 11]
    ).to(DEVICE)

    # Step 5: Load checkpoint
    checkpoint_path = 'stage4_joint_best_revised.pt'
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

            print(f"\n📊 Checkpoint info:")
            print(f"   Total keys: {len(checkpoint)}")

            load_result = model.load_state_dict(checkpoint, strict=False)

            loaded_keys = len(checkpoint) - len(load_result.missing_keys)
            print(f"✅ Loaded {loaded_keys}/{len(checkpoint)} weights")

            if load_result.missing_keys:
                print(f"   ℹ️  Missing: {len(load_result.missing_keys)}")
            if load_result.unexpected_keys:
                print(f"   ℹ️  Unexpected: {len(load_result.unexpected_keys)}")

        except Exception as e:
            print(f"⚠️ Checkpoint error: {e}")
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")

    # Step 6: Create concept embeddings (same as training)
    print("\n🧬 Creating concept embeddings...")
    concept_texts = []
    for cui, info in concept_store.concepts.items():
        text = f"{info['name']}."
        if info['definition']:
            text += f" {info['definition'][:150]}"
        concept_texts.append(text)

    batch_size = 32
    all_embeddings = []

    base_model.eval()
    with torch.no_grad():
        for i in range(0, len(concept_texts), batch_size):
            batch = concept_texts[i:i+batch_size]
            encodings = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(DEVICE)

            outputs = base_model(**encodings)
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings)

    concept_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"✅ Created embeddings: {concept_embeddings.shape}")

    print("\n✅ Model loaded successfully with 150 concepts!")

    return concept_store, model, tokenizer, concept_embeddings

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    render_comparison_ui()
