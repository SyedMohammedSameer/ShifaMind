#!/usr/bin/env python3
"""
ShifaMind Gradio Demo - Interactive Web Interface

Launch with: python gradio_demo.py
Then open browser to http://localhost:7860
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import gradio as gr
import re

# ============================================================================
# CONFIG
# ============================================================================
MODEL_PATH = 'stage4_joint_best_revised.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# ============================================================================
# MODEL COMPONENTS
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

    def forward(self, input_ids, attention_mask, concept_embeddings):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = hidden_states[-1]

        fusion_attentions = []
        for i, fusion_module in enumerate(self.fusion_modules):
            layer_idx = self.fusion_layers[i]
            layer_hidden = hidden_states[layer_idx]
            fused_hidden, attn_weights = fusion_module(layer_hidden, concept_embeddings, attention_mask)
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
print("Loading ShifaMind model...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
concept_cuis = checkpoint['concept_cuis']
num_concepts = checkpoint['num_concepts']
concept_embeddings = checkpoint['concept_embeddings'].to(DEVICE)

base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = ShifaMindModel(
    base_model=base_model,
    num_concepts=num_concepts,
    num_classes=len(TARGET_CODES),
    fusion_layers=[9, 11]
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
print(f"✅ Model loaded: {num_concepts} concepts, {len(TARGET_CODES)} diagnoses")


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================
def predict_diagnosis(clinical_text):
    """Run inference and return formatted results"""

    if not clinical_text.strip():
        return "Please enter clinical text", ""

    # Tokenize
    encoding = tokenizer(
        clinical_text,
        padding='max_length',
        truncation=True,
        max_length=384,
        return_tensors='pt'
    ).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(
            encoding['input_ids'],
            encoding['attention_mask'],
            concept_embeddings
        )

    # Get predictions
    diagnosis_probs = torch.sigmoid(outputs['logits']).cpu().numpy()[0]
    diagnosis_pred_idx = np.argmax(diagnosis_probs)
    diagnosis_code = TARGET_CODES[diagnosis_pred_idx]
    diagnosis_score = float(diagnosis_probs[diagnosis_pred_idx])

    concept_scores = torch.sigmoid(outputs['concept_scores']).cpu().numpy()[0]

    # Get top concepts
    top_concept_indices = np.argsort(concept_scores)[::-1][:5]
    top_concepts = []
    for idx in top_concept_indices:
        if concept_scores[idx] > 0.5:
            top_concepts.append(f"CUI {concept_cuis[idx]}: {concept_scores[idx]:.3f}")

    # Format output
    diagnosis_output = f"""
## 🎯 Diagnosis Prediction

**Code:** {diagnosis_code}
**Description:** {ICD_DESCRIPTIONS[diagnosis_code]}
**Confidence:** {diagnosis_score:.1%}

---

### All Diagnosis Probabilities:
"""
    for code, prob in zip(TARGET_CODES, diagnosis_probs):
        bar = "█" * int(prob * 20)
        diagnosis_output += f"\n**{code}** ({ICD_DESCRIPTIONS[code]}):  \n`{bar}` {prob:.1%}"

    concept_output = f"""
## 💡 Activated Medical Concepts

**Top {len(top_concepts)} Concepts (score > 0.5):**

"""
    if top_concepts:
        for i, concept in enumerate(top_concepts, 1):
            concept_output += f"{i}. {concept}\n"
    else:
        concept_output += "*No concepts activated above threshold*"

    return diagnosis_output, concept_output


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
examples = [
    ["Patient is a 65-year-old male presenting with fever, productive cough, and shortness of breath for 3 days. Chest X-ray shows right lower lobe infiltrate. Vital signs: temp 38.9°C, HR 105, RR 24, BP 135/85. Oxygen saturation 92% on room air. Physical exam reveals crackles in right lower lung field."],
    ["72-year-old female with history of CHF presents with worsening dyspnea and orthopnea. Bilateral lower extremity edema noted. JVD present. Crackles bilateral lung bases. Echo shows EF 25%. BNP elevated at 2400."],
    ["45-year-old male with fever 39.2°C, hypotension 85/50, altered mental status. WBC 18,000. Lactate 4.2. Blood cultures pending. Started on broad-spectrum antibiotics and fluids."],
]

with gr.Blocks(title="ShifaMind - Medical Diagnosis AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 ShifaMind - AI-Powered Medical Diagnosis

    Enter clinical text to get diagnosis predictions with explainable medical concepts.

    **Trained on:** MIMIC-IV discharge summaries
    **Model:** Bio_ClinicalBERT + Concept-Enhanced Architecture
    **F1 Score:** 0.7776 (+8.7% over baseline)
    """)

    with gr.Row():
        with gr.Column(scale=2):
            clinical_input = gr.Textbox(
                label="Clinical Text",
                placeholder="Enter patient presentation, vital signs, symptoms, etc...",
                lines=10
            )
            predict_btn = gr.Button("🔍 Predict Diagnosis", variant="primary", size="lg")

        with gr.Column(scale=3):
            diagnosis_output = gr.Markdown(label="Diagnosis")
            concept_output = gr.Markdown(label="Medical Concepts")

    gr.Examples(
        examples=examples,
        inputs=clinical_input,
        label="Example Cases"
    )

    predict_btn.click(
        fn=predict_diagnosis,
        inputs=clinical_input,
        outputs=[diagnosis_output, concept_output]
    )

    gr.Markdown("""
    ---
    **Note:** This is a research prototype. Not for clinical use.

    **Supported Diagnoses:**
    - J189: Pneumonia, unspecified organism
    - I5023: Acute on chronic systolic heart failure
    - A419: Sepsis, unspecified organism
    - K8000: Calculus of gallbladder with acute cholecystitis
    """)

if __name__ == "__main__":
    demo.launch(share=True)
