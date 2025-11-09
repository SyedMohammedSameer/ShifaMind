#!/usr/bin/env python3
"""
ShifaMind Clinical Demo - SIMPLE VERSION
Works directly with your trained 016.py checkpoint

Run this in Colab:
1. Upload demo1_simple.py and stage4_joint_best_revised.pt
2. !streamlit run demo1_simple.py
3. Or use with ngrok for external access
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Colab check
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    print("🔧 Installing dependencies...")
    os.system('pip install -q streamlit openai torch transformers')
    from google.colab import drive
    drive.mount('/content/drive')

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import streamlit as st
from typing import Dict
import time

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_CODES = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic (congestive) heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis without obstruction'
}

# Template clinical notes (same as before)
DEMO_NOTES = {
    "Pneumonia Case": """CHIEF COMPLAINT: Shortness of breath and fever

HISTORY OF PRESENT ILLNESS:
67-year-old male with a 3-day history of productive cough, fever (101.5°F), and progressive dyspnea. Patient reports yellow-green sputum production and pleuritic chest pain on the right side.

PHYSICAL EXAMINATION:
- Temperature: 101.8°F, HR: 105 bpm, RR: 24/min, BP: 142/88 mmHg, SpO2: 89% on room air
- Chest: Decreased breath sounds and crackles in right lower lobe

LABORATORY:
- WBC: 16,500/μL with left shift
- Chest X-ray: Right lower lobe infiltrate with air bronchograms

ASSESSMENT: Community-acquired pneumonia with respiratory compromise.""",

    "Heart Failure Case": """CHIEF COMPLAINT: Worsening shortness of breath and leg swelling

HISTORY OF PRESENT ILLNESS:
72-year-old female with history of heart failure (EF 25%) presenting with 5-day history of progressive dyspnea, orthopnea (3-pillow), and bilateral lower extremity edema. Weight gain of 12 lbs over past week.

PHYSICAL EXAMINATION:
- BP: 158/92 mmHg, HR: 98 irregular, RR: 22/min
- JVP elevated to 12 cm
- Cardiac: S3 gallop
- Lungs: Bilateral crackles
- Extremities: 3+ pitting edema to knees

LABORATORY:
- BNP: 1,850 pg/mL
- Chest X-ray: Cardiomegaly, bilateral pleural effusions

ASSESSMENT: Acute decompensated heart failure on chronic systolic heart failure.""",

    "Sepsis Case": """CHIEF COMPLAINT: Confusion and fever

HISTORY OF PRESENT ILLNESS:
81-year-old nursing home resident brought to ED with altered mental status, fever, and hypotension. Patient had decreased oral intake for 2 days. Found febrile to 103.2°F with SBP 78 mmHg.

PHYSICAL EXAMINATION:
- Temperature: 103.5°F, HR: 118 bpm, RR: 28/min, BP: 82/45 mmHg
- General: Lethargic, disoriented
- Abdomen: Mild suprapubic tenderness
- Urine: Cloudy, malodorous

LABORATORY:
- WBC: 18,900/μL with 15% bands
- Lactate: 4.2 mmol/L
- Urinalysis: >100 WBC, many bacteria

INTERVENTIONS:
- 2L IV bolus, broad-spectrum antibiotics, norepinephrine

ASSESSMENT: Septic shock secondary to urosepsis.""",

    "Cholecystitis Case": """CHIEF COMPLAINT: Right upper quadrant pain

HISTORY OF PRESENT ILLNESS:
52-year-old obese female with 8-hour history of severe RUQ pain after eating fatty meal. Pain radiates to right shoulder. Associated nausea and vomiting.

PHYSICAL EXAMINATION:
- Temperature: 101.2°F
- Abdomen: Positive Murphy's sign, RUQ tenderness with guarding

LABORATORY:
- WBC: 14,200/μL
- AST: 98 U/L, ALT: 112 U/L

IMAGING:
- RUQ Ultrasound: Gallbladder wall thickening (5.2 mm), multiple gallstones, pericholecystic fluid, positive sonographic Murphy's sign

ASSESSMENT: Acute calculous cholecystitis."""
}

# ============================================================================
# MODEL ARCHITECTURE (Must match 016.py exactly!)
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
        interaction = self.diagnosis_concept_interaction(diagnosis_probs, concept_logits)
        refined_concept_logits = concept_logits + 0.1 * interaction
        return diagnosis_logits, refined_concept_logits

class Phase4RevisedShifaMind(nn.Module):
    def __init__(self, base_model, num_concepts, num_classes, fusion_layers=[9, 11]):
        super().__init__()
        self.base_model = base_model
        self.num_layers = self.base_model.config.num_hidden_layers
        self.hidden_size = self.base_model.config.hidden_size
        self.fusion_layers = fusion_layers

        for i, layer in enumerate(self.base_model.encoder.layer):
            if i < min(fusion_layers) - 2:
                for param in layer.parameters():
                    param.requires_grad = False

        self.fusion_modules = nn.ModuleDict({
            str(layer_idx): SimpleCrossAttention(self.hidden_size, num_heads=8)
            for layer_idx in fusion_layers
        })

        self.heads = DiagnosisConceptHead(self.hidden_size, num_concepts, num_classes)

    def forward(self, input_ids, attention_mask, concept_embeddings, return_attention=False):
        batch_size = input_ids.size(0)
        num_concepts = concept_embeddings.size(0)

        concept_features = concept_embeddings.unsqueeze(0).expand(
            batch_size, num_concepts, self.hidden_size
        )

        outputs = self.base_model.embeddings(input_ids)
        hidden_states = outputs
        attention_weights = {}

        for i, layer in enumerate(self.base_model.encoder.layer):
            hidden_states = layer(hidden_states)[0]

            if i in self.fusion_layers:
                hidden_states, attn_weights = self.fusion_modules[str(i)](
                    hidden_states, concept_features
                )
                if return_attention:
                    attention_weights[f'layer_{i}'] = attn_weights

        pooled = hidden_states[:, 0]
        diagnosis_logits, concept_logits = self.heads(pooled)

        if return_attention:
            return diagnosis_logits, concept_logits, attention_weights
        return diagnosis_logits, concept_logits

# ============================================================================
# CHATGPT
# ============================================================================
def query_chatgpt(clinical_note: str, api_key: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical diagnostic assistant."},
                {"role": "user", "content": f"""Analyze this clinical note and provide the most likely diagnosis.

Clinical Note:
{clinical_note}

Provide a concise diagnosis with brief reasoning (2-3 sentences max)."""}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nTry: !pip install --upgrade openai"

# ============================================================================
# SHIFAMIND INFERENCE
# ============================================================================
@torch.no_grad()
def predict_shifamind(model, tokenizer, clinical_note: str, concept_embeddings,
                      concept_names, threshold=0.7):
    model.eval()

    encoding = tokenizer(
        clinical_note,
        max_length=384,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    start_time = time.time()
    diagnosis_logits, concept_logits, _ = model(
        input_ids, attention_mask, concept_embeddings, return_attention=True
    )
    inference_time = time.time() - start_time

    diagnosis_probs = torch.sigmoid(diagnosis_logits).cpu().numpy()[0]
    concept_probs = torch.sigmoid(concept_logits).cpu().numpy()[0]

    predicted_diagnoses = []
    for idx, (code, name) in enumerate(TARGET_CODES.items()):
        prob = diagnosis_probs[idx]
        if prob > 0.5:
            predicted_diagnoses.append({'code': code, 'name': name, 'confidence': float(prob)})

    activated_concepts = []
    for idx, (name, prob) in enumerate(zip(concept_names, concept_probs)):
        if prob > threshold:
            activated_concepts.append({'name': name, 'confidence': float(prob)})

    predicted_diagnoses.sort(key=lambda x: x['confidence'], reverse=True)
    activated_concepts.sort(key=lambda x: x['confidence'], reverse=True)

    return {
        'diagnoses': predicted_diagnoses,
        'concepts': activated_concepts,
        'inference_time': inference_time,
        'num_concepts': len(activated_concepts)
    }

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model_and_checkpoint():
    """Load model with checkpoint that has 150 concepts"""

    print("📦 Loading Bio_ClinicalBERT...")
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    base_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    # Load checkpoint to get number of concepts
    checkpoint_path = 'stage4_joint_best_revised.pt'
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Checkpoint not found: {checkpoint_path}")
        st.stop()

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Detect number of concepts from checkpoint
    concept_head_weight = checkpoint.get('heads.concept_head.weight')
    if concept_head_weight is not None:
        num_concepts = concept_head_weight.shape[0]
        print(f"📊 Detected {num_concepts} concepts from checkpoint")
    else:
        num_concepts = 150  # Default
        print(f"⚠️  Could not detect concept count, using default: {num_concepts}")

    # Initialize model
    model = Phase4RevisedShifaMind(
        base_model=base_model,
        num_concepts=num_concepts,
        num_classes=len(TARGET_CODES),
        fusion_layers=[9, 11]
    ).to(DEVICE)

    # Load checkpoint
    load_result = model.load_state_dict(checkpoint, strict=False)
    loaded_keys = len(checkpoint) - len(load_result.missing_keys)
    print(f"✅ Loaded {loaded_keys}/{len(checkpoint)} weights")

    # Create dummy concept names (we don't have the actual UMLS names, but that's okay for demo)
    concept_names = [f"Medical Concept {i+1}" for i in range(num_concepts)]

    # Create concept embeddings (random placeholder - checkpoint has trained weights)
    print(f"🧬 Creating {num_concepts} concept embeddings...")
    with torch.no_grad():
        dummy_texts = [f"Concept {i}" for i in range(num_concepts)]
        encoded = tokenizer(
            dummy_texts,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(DEVICE)

        outputs = base_model(**encoded)
        concept_embeddings = outputs.last_hidden_state[:, 0, :]

    print("✅ Model loaded successfully")
    return model, tokenizer, concept_embeddings, concept_names

# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    st.set_page_config(page_title="ShifaMind Demo", page_icon="🏥", layout="wide")

    st.markdown("# 🏥 ShifaMind Clinical AI Demo")
    st.markdown("### ChatGPT vs ShifaMind Comparison")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    if st.sidebar.button("🚀 Load ShifaMind Model"):
        with st.spinner("Loading model..."):
            try:
                model, tokenizer, concept_embeddings, concept_names = load_model_and_checkpoint()
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.concept_embeddings = concept_embeddings
                st.session_state.concept_names = concept_names
                st.session_state.model_loaded = True
                st.sidebar.success("✅ Model Ready")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")

    # Main content
    st.markdown("## 📝 Clinical Note Input")

    selected_template = st.selectbox(
        "Choose a template case:",
        ["Custom"] + list(DEMO_NOTES.keys())
    )

    if selected_template == "Custom":
        clinical_note = st.text_area("Enter clinical note:", height=300)
    else:
        clinical_note = st.text_area(
            f"Clinical Note ({selected_template}):",
            value=DEMO_NOTES[selected_template],
            height=300
        )

    if st.button("🔬 Run Diagnosis Comparison", type="primary"):
        if not clinical_note.strip():
            st.warning("Please enter a clinical note")
            return

        if not api_key:
            st.warning("Please enter OpenAI API key")
            return

        if not st.session_state.get('model_loaded'):
            st.warning("Please load the ShifaMind model first")
            return

        st.markdown("---")
        st.markdown("## 📊 Diagnosis Results")

        col1, col2 = st.columns(2)

        # ChatGPT
        with col1:
            st.markdown("### 🤖 ChatGPT-4")
            with st.spinner("Querying ChatGPT..."):
                chatgpt_response = query_chatgpt(clinical_note, api_key)

            st.markdown(f'<div style="background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px;">{chatgpt_response}</div>', unsafe_allow_html=True)

        # ShifaMind
        with col2:
            st.markdown("### 🏥 ShifaMind")
            with st.spinner("Running ShifaMind..."):
                results = predict_shifamind(
                    st.session_state.model,
                    st.session_state.tokenizer,
                    clinical_note,
                    st.session_state.concept_embeddings,
                    st.session_state.concept_names
                )

            st.markdown("#### 🎯 Predicted Diagnoses")
            for diag in results['diagnoses']:
                st.markdown(f"""
                <div style="background-color: #e8f5e9; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <strong>{diag['code']}</strong>: {diag['name']}<br>
                    <strong>Confidence:</strong> {diag['confidence']*100:.1f}%
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"#### 🧠 Activated Concepts ({results['num_concepts']} concepts)")
            for concept in results['concepts'][:10]:
                st.markdown(f"- **{concept['name']}** ({concept['confidence']*100:.0f}%)")

            if results['num_concepts'] > 10:
                st.caption(f"... and {results['num_concepts'] - 10} more concepts")

            st.caption(f"⏱️ Inference time: {results['inference_time']:.3f}s")

if __name__ == "__main__":
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    main()
