#!/usr/bin/env python3
"""
ShifaMind 042: Interactive Gradio Demo

Interactive web interface for explainable clinical diagnosis prediction.

Author: Mohammed Sameer Syed
Date: November 2025
Version: 042-Demo
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
CHECKPOINT_PATH = BASE_PATH / '03_Models/checkpoints'
CHECKPOINT_FINAL = CHECKPOINT_PATH / 'shifamind_042_final.pt'

TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# ============================================================================
# MODEL ARCHITECTURE (COPY FROM 042.py)
# ============================================================================

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
        self.diagnosis_concept_interaction = nn.Bilinear(num_classes, num_concepts, num_concepts)
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


# ============================================================================
# LOAD MODEL
# ============================================================================

print("="*80)
print("SHIFAMIND 042: INTERACTIVE DEMO")
print("="*80)

print("\n📂 Loading model...")

# Check if checkpoint exists
if not CHECKPOINT_FINAL.exists():
    print(f"❌ ERROR: Checkpoint not found at {CHECKPOINT_FINAL}")
    print("   Please run 042.py first to train the model.")
    sys.exit(1)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_FINAL, map_location=device)
concept_embeddings = checkpoint['concept_embeddings'].to(device)
num_concepts = checkpoint['num_concepts']
concept_cuis = checkpoint['concept_cuis']
concept_names = checkpoint['concept_names']

print(f"  ✅ Loaded {num_concepts} concept embeddings")

# Initialize tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

# Initialize ShifaMind model
model = ShifaMindModel(
    base_model=base_model,
    num_concepts=num_concepts,
    num_classes=len(TARGET_CODES),
    fusion_layers=[9, 11]
).to(device)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Build concept store
concept_store = {
    cui: {'preferred_name': name, 'semantic_types': []}
    for cui, name in concept_names.items()
}
concept_to_idx = {cui: i for i, cui in enumerate(concept_cuis)}
idx_to_concept = {i: cui for i, cui in enumerate(concept_cuis)}

print(f"  ✅ Model loaded successfully!")
print("="*80)

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_with_explanation(clinical_note):
    """
    Input: Clinical note
    Output: Diagnosis + reasoning chain
    """

    if not clinical_note or len(clinical_note.strip()) < 10:
        return (
            "⚠️ Please enter a valid clinical note (at least 10 characters)",
            "",
            "No analysis performed",
            "{}"
        )

    try:
        # Tokenize input
        encoding = tokenizer(
            clinical_note,
            padding='max_length',
            truncation=True,
            max_length=384,
            return_tensors='pt'
        ).to(device)

        # Get prediction
        model.eval()
        with torch.no_grad():
            outputs = model(
                encoding['input_ids'],
                encoding['attention_mask'],
                concept_embeddings
            )

            # Get diagnosis probabilities
            diagnosis_probs = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
            concept_scores = torch.sigmoid(outputs['concept_scores']).cpu().numpy()[0]

        # Get top predicted diagnosis
        pred_idx = np.argmax(diagnosis_probs)
        pred_code = TARGET_CODES[pred_idx]
        pred_conf = diagnosis_probs[pred_idx]
        pred_name = ICD_DESCRIPTIONS[pred_code]

        # Get top concepts
        top_k = 10
        top_indices = np.argsort(concept_scores)[::-1][:top_k]

        concepts = []
        for idx in top_indices:
            if concept_scores[idx] > 0.3:  # Threshold
                cui = idx_to_concept.get(idx, f"CUI_{idx}")
                concept_info = concept_store.get(cui, {})
                concepts.append({
                    'cui': cui,
                    'name': concept_info.get('preferred_name', f'Concept_{idx}'),
                    'score': float(concept_scores[idx])
                })

        # Format output
        diagnosis_text = f"**{pred_name}** ({pred_code})"

        confidence_text = f"{pred_conf:.1%}"

        # Create reasoning markdown
        reasoning_lines = []
        reasoning_lines.append(f"## Diagnosis: {pred_name}")
        reasoning_lines.append(f"**Code:** {pred_code}")
        reasoning_lines.append(f"**Confidence:** {pred_conf:.1%}")
        reasoning_lines.append("")
        reasoning_lines.append("### Supporting Concepts:")
        reasoning_lines.append("")

        for i, concept in enumerate(concepts, 1):
            reasoning_lines.append(
                f"{i}. **{concept['name']}** (score: {concept['score']:.1%})"
            )

        reasoning_lines.append("")
        reasoning_lines.append(f"*Total concepts: {len(concepts)}*")

        # All diagnosis probabilities
        reasoning_lines.append("")
        reasoning_lines.append("### All Diagnosis Probabilities:")
        reasoning_lines.append("")
        for i, code in enumerate(TARGET_CODES):
            prob = diagnosis_probs[i]
            bar = "█" * int(prob * 20)
            reasoning_lines.append(f"- **{code}** ({ICD_DESCRIPTIONS[code][:40]}...): {prob:.1%} {bar}")

        reasoning_markdown = "\n".join(reasoning_lines)

        # Create JSON output
        json_output = {
            "diagnosis": {
                "code": pred_code,
                "name": pred_name,
                "confidence": float(pred_conf)
            },
            "all_probabilities": {
                code: float(diagnosis_probs[i])
                for i, code in enumerate(TARGET_CODES)
            },
            "reasoning_chain": [
                {
                    "concept": c['name'],
                    "cui": c['cui'],
                    "score": float(c['score'])
                }
                for c in concepts
            ],
            "num_concepts": len(concepts),
            "metadata": {
                "model_version": "ShifaMind-042",
                "timestamp": datetime.now().isoformat(),
                "note_length": len(clinical_note.split())
            }
        }

        json_output_str = json.dumps(json_output, indent=2)

        return diagnosis_text, confidence_text, reasoning_markdown, json_output_str

    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        return error_msg, "", "Error occurred", "{}"


# ============================================================================
# EXAMPLE CLINICAL NOTES
# ============================================================================

EXAMPLE_PNEUMONIA = """
CHIEF COMPLAINT: Fever, cough, and shortness of breath

HISTORY OF PRESENT ILLNESS:
72-year-old male presents with 3-day history of productive cough with yellow sputum,
fever up to 101.5°F, and progressive dyspnea. Patient reports chills and fatigue.
Denies recent travel or sick contacts.

PHYSICAL EXAMINATION:
Vital Signs: T 101.2°F, HR 98, BP 130/80, RR 22, SpO2 92% on room air
Lungs: Decreased breath sounds at right base with crackles. Dullness to percussion.
Heart: Regular rate and rhythm, no murmurs

IMAGING:
Chest X-ray shows right lower lobe consolidation consistent with pneumonia.

LABS:
WBC 14,500 with left shift
"""

EXAMPLE_HEART_FAILURE = """
CHIEF COMPLAINT: Worsening shortness of breath and leg swelling

HISTORY OF PRESENT ILLNESS:
68-year-old female with history of heart failure presents with progressive dyspnea
over past week. Reports orthopnea requiring 3 pillows, paroxysmal nocturnal dyspnea,
and bilateral lower extremity edema. Also notes increased fatigue and decreased
exercise tolerance.

PHYSICAL EXAMINATION:
Vital Signs: BP 145/90, HR 88, RR 20, SpO2 94% on 2L O2
Cardiovascular: S3 gallop present, JVP elevated at 12 cm
Lungs: Bilateral crackles at bases
Extremities: 3+ pitting edema bilateral lower extremities

LABS:
BNP 1850 pg/mL (elevated)
Creatinine 1.4 mg/dL

IMAGING:
Chest X-ray shows cardiomegaly and pulmonary edema
"""

EXAMPLE_SEPSIS = """
CHIEF COMPLAINT: Fever, confusion, and hypotension

HISTORY OF PRESENT ILLNESS:
58-year-old male brought to ED by family for altered mental status. Patient has been
confused for past 24 hours, with fever and chills. Family notes patient has had
decreased urine output. Patient appears lethargic and disoriented.

PHYSICAL EXAMINATION:
Vital Signs: T 102.8°F, HR 115, BP 85/55, RR 24, SpO2 93%
General: Lethargic, disoriented to time and place
Skin: Warm, flushed

LABS:
WBC 18,000 with 15% bands
Lactate 4.5 mmol/L (elevated)
Creatinine 2.1 mg/dL (acute kidney injury)
Blood cultures pending

ASSESSMENT: Sepsis with hypotension and organ dysfunction
"""

EXAMPLE_CHOLECYSTITIS = """
CHIEF COMPLAINT: Right upper quadrant abdominal pain

HISTORY OF PRESENT ILLNESS:
55-year-old female presents with acute onset right upper quadrant pain that started
after eating a fatty meal. Pain is constant, severe (8/10), and radiates to right
shoulder. Patient reports nausea and one episode of vomiting. Has had similar but
milder episodes in past.

PHYSICAL EXAMINATION:
Vital Signs: T 100.8°F, HR 92, BP 135/82, RR 18
Abdomen: Tender in right upper quadrant with positive Murphy's sign
No rebound or guarding

LABS:
WBC 13,200
Mild elevation in liver enzymes

IMAGING:
Ultrasound shows gallbladder wall thickening (5mm), pericholecystic fluid,
and multiple gallstones. Positive sonographic Murphy's sign.

IMPRESSION: Acute cholecystitis
"""

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

demo = gr.Interface(
    fn=predict_with_explanation,
    inputs=gr.Textbox(
        label="Clinical Note",
        placeholder="Enter discharge summary or clinical note here...",
        lines=15
    ),
    outputs=[
        gr.Textbox(label="🎯 Predicted Diagnosis", lines=1),
        gr.Textbox(label="📊 Confidence", lines=1),
        gr.Markdown(label="🔍 Reasoning Chain & Analysis"),
        gr.Code(label="📄 JSON Output", language="json", lines=15)
    ],
    title="🏥 ShifaMind: Explainable Clinical AI",
    description="""
    **ShifaMind 042** provides explainable diagnosis predictions for clinical notes.

    Enter a clinical note below and receive:
    - Predicted diagnosis with confidence
    - Reasoning chain showing supporting medical concepts
    - Structured JSON output for integration

    **Supported Diagnoses:**
    - J189: Pneumonia, unspecified organism
    - I5023: Acute on chronic systolic heart failure
    - A419: Sepsis, unspecified organism
    - K8000: Acute cholecystitis with gallstones
    """,
    examples=[
        [EXAMPLE_PNEUMONIA],
        [EXAMPLE_HEART_FAILURE],
        [EXAMPLE_SEPSIS],
        [EXAMPLE_CHOLECYSTITIS]
    ],
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    """,
    allow_flagging="never"
)

if __name__ == '__main__':
    print("\n🚀 Launching Gradio demo...")
    print("   Access the interface in your browser")
    print("="*80)
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
