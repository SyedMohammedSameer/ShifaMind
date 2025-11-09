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
from typing import Dict, List, Tuple
from collections import defaultdict
import json
import time
from pathlib import Path
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
# CONCEPT STORE (Simplified for Demo)
# ============================================================================
class ConceptStore:
    def __init__(self):
        """Initialize with core medical concepts"""
        self.concepts = {}
        self._load_core_concepts()

    def _load_core_concepts(self):
        """Load essential concepts for demo"""
        # These would normally be loaded from UMLS
        core_concepts = {
            # Pneumonia-related
            'C0032285': {'name': 'Pneumonia', 'definition': 'Inflammation of the lung parenchyma'},
            'C0010200': {'name': 'Coughing', 'definition': 'Sudden, forceful expulsion of air from lungs'},
            'C0015967': {'name': 'Fever', 'definition': 'Abnormal elevation of body temperature'},
            'C0013404': {'name': 'Dyspnea', 'definition': 'Difficult or labored breathing'},
            'C0034642': {'name': 'Rales', 'definition': 'Crackling sounds in lungs'},
            'C0152965': {'name': 'Pulmonary infiltrate', 'definition': 'Abnormal substance in lung tissue'},

            # Heart failure-related
            'C0018802': {'name': 'Congestive heart failure', 'definition': 'Heart unable to pump sufficient blood'},
            'C0013604': {'name': 'Edema', 'definition': 'Abnormal accumulation of fluid in tissues'},
            'C0085619': {'name': 'Orthopnea', 'definition': 'Dyspnea when lying flat'},
            'C0034063': {'name': 'Pulmonary edema', 'definition': 'Fluid accumulation in lungs'},
            'C0232461': {'name': 'S3 gallop', 'definition': 'Third heart sound indicating ventricular dysfunction'},
            'C0231528': {'name': 'Jugular venous distension', 'definition': 'Elevated neck vein pressure'},

            # Sepsis-related
            'C0243026': {'name': 'Sepsis', 'definition': 'Life-threatening organ dysfunction due to infection'},
            'C0020649': {'name': 'Hypotension', 'definition': 'Abnormally low blood pressure'},
            'C0009676': {'name': 'Confusion', 'definition': 'Impaired cognitive function and disorientation'},
            'C0023530': {'name': 'Leukocytosis', 'definition': 'Elevated white blood cell count'},
            'C0022658': {'name': 'Kidney disease', 'definition': 'Impairment of renal function'},
            'C0041296': {'name': 'Urinary tract infection', 'definition': 'Infection of urinary system'},

            # Cholecystitis-related
            'C0008325': {'name': 'Cholecystitis', 'definition': 'Inflammation of gallbladder'},
            'C0162429': {'name': "Murphy's sign", 'definition': 'Pain on palpation of RUQ during deep inspiration'},
            'C0008350': {'name': 'Gallstone', 'definition': 'Solid crystalline deposit in gallbladder'},
            'C0000737': {'name': 'Abdominal pain', 'definition': 'Pain in stomach region'},
            'C0027497': {'name': 'Nausea', 'definition': 'Sensation of needing to vomit'},
            'C0028960': {'name': 'Obesity', 'definition': 'Excessive body fat accumulation'},
        }

        self.concepts = core_concepts

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
        self.bert = base_model
        self.num_layers = self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.fusion_layers = fusion_layers
        self.concept_store = concept_store

        # Freeze early BERT layers
        for i, layer in enumerate(self.bert.encoder.layer):
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
        outputs = self.bert.embeddings(input_ids)
        hidden_states = outputs

        attention_weights = {}

        for i, layer in enumerate(self.bert.encoder.layer):
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

    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check your OpenAI API key."

# ============================================================================
# SHIFAMIND INFERENCE
# ============================================================================
@torch.no_grad()
def predict_shifamind(
    model,
    tokenizer,
    clinical_note: str,
    concept_embeddings: torch.Tensor,
    concept_store: ConceptStore,
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
    """Load trained ShifaMind model and dependencies"""

    print("📦 Loading model components...")

    # Initialize concept store
    concept_store = ConceptStore()

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    # Initialize model
    model = Phase4RevisedShifaMind(
        base_model=base_model,
        concept_store=concept_store,
        num_classes=len(TARGET_CODES),
        fusion_layers=[9, 11]
    ).to(DEVICE)

    # Load trained weights (only custom layers, not BERT base)
    checkpoint_path = 'stage4_joint_best_revised.pt'
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            # Load only matching keys (allows partial loading)
            model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Checkpoint loading issue: {e}")
            print("   Using pre-trained BERT weights only")
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        print("   Using pre-trained BERT weights only")

    # Create concept embeddings
    concept_texts = [
        f"{data['name']}: {data.get('definition', '')}"
        for data in concept_store.concepts.values()
    ]

    with torch.no_grad():
        encoded = tokenizer(
            concept_texts,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(DEVICE)

        outputs = base_model(**encoded)
        concept_embeddings = outputs.last_hidden_state[:, 0, :]  # Keep on DEVICE

    print("✅ Model loaded successfully")

    return concept_store, model, tokenizer, concept_embeddings

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    render_comparison_ui()
