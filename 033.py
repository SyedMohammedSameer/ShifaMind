#!/usr/bin/env python3
"""
ShifaMind Demo Script

Loads trained ShifaMind model from 032.py and runs inference.

Usage:
  python 033.py

Outputs:
  - Diagnosis predictions with confidence scores
  - Activated medical concepts (by CUI)
  - Evidence spans from clinical text supporting each concept
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = 'stage4_joint_best_revised.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Target diagnoses
TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# ============================================================================
# MODEL COMPONENTS (copied from 032.py)
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


class EvidenceExtractor:
    """Extracts evidence spans from clinical text using attention weights"""

    def __init__(self, tokenizer,
                 attention_percentile=85,
                 min_span_tokens=5,
                 max_span_tokens=50,
                 top_k_spans=3):
        self.tokenizer = tokenizer
        self.attention_percentile = attention_percentile
        self.min_span_tokens = min_span_tokens
        self.max_span_tokens = max_span_tokens
        self.top_k_spans = top_k_spans

    def extract(self, input_ids, attention_weights, concept_scores, concept_cuis, threshold=0.7):
        """Extract evidence spans for activated concepts"""

        # Get activated concepts
        activated = []
        for idx, score in enumerate(concept_scores):
            if score > threshold and idx < len(concept_cuis):
                activated.append({
                    'idx': idx,
                    'cui': concept_cuis[idx],
                    'score': float(score)
                })

        activated = sorted(activated, key=lambda x: x['score'], reverse=True)[:5]

        if not activated:
            return []

        # Aggregate attention across layers
        aggregated_attention = torch.stack(attention_weights).mean(dim=0)

        results = []
        for concept_info in activated:
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

                # Filter EHR noise
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
        """Check if text is EHR noise (headers, templates, etc)"""
        noise_patterns = [
            r'_{3,}',  # Redaction markers
            r'^\s*(name|unit no|admission date|discharge date|date of birth|sex|service|allergies|attending|chief complaint|major surgical)\s*:',
            r'\[\s*\*\*.*?\*\*\s*\]',  # De-ID markers
            r'^[^a-z]{10,}$',  # No lowercase (headers)
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in noise_patterns)


# ============================================================================
# DEMO
# ============================================================================
def load_model():
    """Load trained ShifaMind model"""
    print("Loading model...")

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Get metadata
    concept_cuis = checkpoint['concept_cuis']
    num_concepts = checkpoint['num_concepts']

    print(f"✅ Model loaded: {num_concepts} concepts, 4 diagnoses")

    # Initialize model architecture
    base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = ShifaMindModel(
        base_model=base_model,
        num_concepts=num_concepts,
        num_classes=len(TARGET_CODES),
        fusion_layers=[9, 11]
    )

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # Get concept embeddings (saved in checkpoint by 032.py)
    if 'concept_embeddings' in checkpoint:
        concept_embeddings = checkpoint['concept_embeddings'].to(DEVICE)
    else:
        print("⚠️  Warning: concept_embeddings not in checkpoint, model may not work correctly")
        concept_embeddings = None

    return model, concept_cuis, concept_embeddings


def predict(clinical_text, model, concept_cuis, concept_embeddings, tokenizer):
    """Run inference on clinical text"""

    # Tokenize
    encoding = tokenizer(
        clinical_text,
        padding='max_length',
        truncation=True,
        max_length=384,
        return_tensors='pt'
    ).to(DEVICE)

    # Run model
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

    # Extract evidence
    evidence_extractor = EvidenceExtractor(tokenizer)
    sample_attention_weights = [attn[0] for attn in outputs['attention_weights']]
    evidence = evidence_extractor.extract(
        encoding['input_ids'][0].cpu(),
        sample_attention_weights,
        concept_scores,
        concept_cuis
    )

    return {
        'diagnosis_code': diagnosis_code,
        'diagnosis_name': ICD_DESCRIPTIONS[diagnosis_code],
        'confidence': diagnosis_score,
        'evidence': evidence
    }


def print_results(result):
    """Pretty print results"""
    print("\n" + "="*70)
    print("SHIFAMIND PREDICTION")
    print("="*70)

    print(f"\n🎯 Diagnosis: {result['diagnosis_code']} - {result['diagnosis_name']}")
    print(f"   Confidence: {result['confidence']:.3f}")

    print(f"\n💡 Activated Concepts ({len(result['evidence'])} total):")
    for i, extraction in enumerate(result['evidence'], 1):
        print(f"\n  [{i}] CUI: {extraction['concept_cui']}")
        print(f"      Concept Score: {extraction['concept_score']:.3f}")
        print(f"      Evidence Spans ({len(extraction['evidence_spans'])} found):")

        for j, span in enumerate(extraction['evidence_spans'], 1):
            print(f"\n      Span {j} (attention: {span['attention_score']:.3f}):")
            print(f"      \"{span['text']}\"")

    print("\n" + "="*70)


def main():
    """Run demo"""
    print("ShifaMind Demo - Medical Diagnosis with Explainability\n")

    # Load
    model, concept_cuis, concept_embeddings = load_model()

    if concept_embeddings is None:
        print("\n❌ Error: Concept embeddings not found in checkpoint.")
        print("Make sure you're using a checkpoint from 032.py that includes concept_embeddings.")
        return

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Example clinical text
    clinical_text = """
    Patient is a 65-year-old male presenting with fever, productive cough,
    and shortness of breath for 3 days. Chest X-ray shows right lower lobe
    infiltrate. Vital signs: temp 38.9°C, HR 105, RR 24, BP 135/85.
    Oxygen saturation 92% on room air. Physical exam reveals crackles
    in right lower lung field. Started on empiric antibiotics.
    """

    print("Clinical Text:")
    print(clinical_text)

    # Predict
    result = predict(clinical_text, model, concept_cuis, concept_embeddings, tokenizer)

    # Display
    print_results(result)


if __name__ == "__main__":
    main()
