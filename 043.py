#!/usr/bin/env python3
"""
ShifaMind 043: Evidence Extraction & Knowledge Retrieval (Phase 2)

Loads 042 checkpoint and adds:
1. Attention weight capture for evidence extraction
2. Evidence span extraction from clinical notes
3. Integration with clinical knowledge base

CRITICAL: This is a CODE-ONLY modification (no new parameters, no retraining needed)

Author: Mohammed Sameer Syed
Date: November 2025
Version: 043-Evidence
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
import math

# Seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

# ============================================================================
# 2. CONFIGURATION
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
CHECKPOINT_PATH = BASE_PATH / '03_Models/checkpoints'
CHECKPOINT_042 = CHECKPOINT_PATH / 'shifamind_042_final.pt'
CHECKPOINT_043 = CHECKPOINT_PATH / 'shifamind_043_final.pt'

TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

print("="*80)
print("SHIFAMIND 043: EVIDENCE EXTRACTION & KNOWLEDGE RETRIEVAL")
print("="*80)
print(f"\n📁 Loading from: {CHECKPOINT_042}")
print(f"📁 Saving to: {CHECKPOINT_043}")

# ============================================================================
# 3. MODIFIED MODEL ARCHITECTURE WITH ATTENTION CAPTURE
# ============================================================================

class EnhancedCrossAttention(nn.Module):
    """
    Cross-attention with optional attention weight return

    MODIFICATION: Added return_attention parameter to capture attention weights
    for evidence extraction (no new parameters, code-only change)
    """

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

    def forward(self, hidden_states, concept_embeddings, attention_mask=None, return_attention=False):
        """
        Forward pass with optional attention weight return

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            concept_embeddings: [num_concepts, hidden_size]
            attention_mask: [batch, seq_len] (optional)
            return_attention: If True, return attention weights

        Returns:
            If return_attention=False: output [batch, seq_len, hidden_size]
            If return_attention=True: (output, attention_weights [batch, num_heads, seq_len, num_concepts])
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_concepts = concept_embeddings.shape[0]

        # Expand concepts to batch
        concepts_batch = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Multi-head projections
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, num_concepts]
        attention_weights_dropout = self.dropout(attention_weights)

        # Weighted sum
        context = torch.matmul(attention_weights_dropout, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        # Gated fusion
        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        output = hidden_states + gate_values * context
        output = self.layer_norm(output)

        if return_attention:
            return output, attention_weights  # Return WITHOUT dropout for cleaner visualization
        return output


class ShifaMindModel043(nn.Module):
    """
    ShifaMind Phase 2 model with evidence extraction capability

    MODIFICATION: Added return_attention parameter to capture attention weights
    from layers 9 and 11 for evidence extraction
    """

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

    def forward(self, input_ids, attention_mask, concept_embeddings, return_diagnosis_only=False, return_attention=False):
        """
        Forward pass with optional attention weight capture

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            concept_embeddings: [num_concepts, hidden_size]
            return_diagnosis_only: If True, only return diagnosis logits
            return_attention: If True, return attention weights for evidence extraction

        Returns:
            If return_attention=False: {'logits': ..., 'concept_scores': ...}
            If return_attention=True: {'logits': ..., 'concept_scores': ..., 'attention_weights': {...}}
        """
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

        fusion_attentions = {}

        for i, fusion_module in enumerate(self.fusion_modules):
            layer_idx = self.fusion_layers[i]
            layer_hidden = hidden_states[layer_idx]

            if return_attention:
                fused_hidden, attn_weights = fusion_module(
                    layer_hidden, concept_embeddings, attention_mask, return_attention=True
                )
                fusion_attentions[f'layer_{layer_idx}'] = attn_weights
            else:
                fused_hidden = fusion_module(
                    layer_hidden, concept_embeddings, attention_mask, return_attention=False
                )

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

        result = {
            'logits': diagnosis_logits,
            'concept_scores': refined_concept_logits,
        }

        if return_attention:
            fusion_attentions['input_ids'] = input_ids
            result['attention_weights'] = fusion_attentions

        return result


# ============================================================================
# 4. EVIDENCE EXTRACTION FUNCTION
# ============================================================================

def extract_evidence_spans(
    text: str,
    input_ids: torch.Tensor,
    attention_weights: Dict,
    concepts: List[Dict],
    tokenizer,
    top_k: int = 5,
    span_window: int = 5
) -> List[Dict]:
    """
    Extract evidence text spans using attention weights

    Args:
        text: Original input text (string)
        input_ids: Token IDs [seq_len]
        attention_weights: Dict with 'layer_9', 'layer_11' attention tensors
        concepts: List of concept dicts with 'name', 'cui', 'score', 'idx'
        tokenizer: BioClinicalBERT tokenizer
        top_k: Number of concepts to extract evidence for
        span_window: Context window around high-attention tokens

    Returns:
        List of dicts with concept and evidence spans
    """

    # Average attention across layers and heads
    # attention_weights['layer_9']: [1, num_heads, seq_len, num_concepts]
    # attention_weights['layer_11']: [1, num_heads, seq_len, num_concepts]

    layer_9_attn = attention_weights.get('layer_9')
    layer_11_attn = attention_weights.get('layer_11')

    if layer_9_attn is None or layer_11_attn is None:
        return []

    # Average across heads and layers
    attn_9 = layer_9_attn.squeeze(0).mean(0)  # [seq_len, num_concepts]
    attn_11 = layer_11_attn.squeeze(0).mean(0)  # [seq_len, num_concepts]
    avg_attention = (attn_9 + attn_11) / 2  # [seq_len, num_concepts]

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().tolist())

    evidence_chains = []

    for concept in concepts[:top_k]:
        concept_idx = concept.get('idx', 0)

        if concept_idx >= avg_attention.shape[1]:
            continue

        # Get attention for this concept
        concept_attention = avg_attention[:, concept_idx]  # [seq_len]

        # Find top-5 attended tokens
        topk_values, topk_indices = torch.topk(
            concept_attention,
            k=min(5, len(tokens))
        )

        # Extract spans around high-attention tokens
        spans = []
        for token_idx in topk_indices:
            token_idx = token_idx.item()

            # Get surrounding context
            start = max(0, token_idx - span_window)
            end = min(len(tokens), token_idx + span_window + 1)

            # Extract token span
            span_tokens = tokens[start:end]

            # Convert to text (handle wordpiece tokens)
            span_text = tokenizer.convert_tokens_to_string(span_tokens)

            # Clean up
            span_text = span_text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').strip()

            if len(span_text) > 10:  # Only keep meaningful spans
                spans.append(span_text)

        # Remove duplicates and limit to top 3
        unique_spans = []
        seen = set()
        for span in spans:
            span_lower = span.lower()
            if span_lower not in seen:
                unique_spans.append(span)
                seen.add(span_lower)
                if len(unique_spans) >= 3:
                    break

        evidence_chains.append({
            'concept': concept['name'],
            'cui': concept.get('cui', 'UNKNOWN'),
            'score': float(concept['score']),
            'evidence_spans': unique_spans[:3],
            'semantic_types': concept.get('semantic_types', [])
        })

    return evidence_chains


# ============================================================================
# 5. LOAD 042 CHECKPOINT AND CREATE 043
# ============================================================================

def main():
    """
    Load 042 checkpoint, add evidence extraction, test, and save as 043
    """

    print("\n" + "="*80)
    print("PHASE 2: ADDING EVIDENCE EXTRACTION TO TRAINED MODEL")
    print("="*80)

    # Check if 042 checkpoint exists
    if not CHECKPOINT_042.exists():
        print(f"\n❌ ERROR: 042 checkpoint not found at {CHECKPOINT_042}")
        print("   Please run 042.py first to train the model.")
        return

    print(f"\n📂 Loading 042 checkpoint...")
    checkpoint = torch.load(CHECKPOINT_042, map_location=device)

    # Extract checkpoint data
    concept_embeddings = checkpoint['concept_embeddings'].to(device)
    num_concepts = checkpoint['num_concepts']
    concept_cuis = checkpoint['concept_cuis']
    concept_names = checkpoint['concept_names']

    print(f"  ✅ Loaded {num_concepts} concept embeddings")

    # Build concept store for evidence extraction
    concept_store = {
        'concepts': {
            cui: {'preferred_name': name, 'semantic_types': []}
            for cui, name in concept_names.items()
        },
        'concept_to_idx': {cui: i for i, cui in enumerate(concept_cuis)},
        'idx_to_concept': {i: cui for i, cui in enumerate(concept_cuis)}
    }

    # Initialize tokenizer and base model
    print(f"\n📦 Initializing Bio_ClinicalBERT...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

    # Initialize ShifaMind 043 model
    print(f"\n🏗️  Initializing ShifaMind043 model...")
    model = ShifaMindModel043(
        base_model=base_model,
        num_concepts=num_concepts,
        num_classes=len(TARGET_CODES),
        fusion_layers=[9, 11]
    ).to(device)

    # Load 042 weights
    print(f"\n⚙️  Loading 042 trained weights...")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    print(f"  ✅ Model loaded successfully!")

    # Test evidence extraction on sample text
    print(f"\n🧪 Testing evidence extraction...")

    test_text = """
    Patient presents with fever of 38.9°C for 3 days, productive cough with yellow sputum,
    and progressive shortness of breath. Physical examination reveals crackles in the right
    lower lobe and decreased breath sounds. Chest X-ray shows right lower lobe infiltrate
    consistent with pneumonia. White blood cell count is elevated at 14,500.
    """

    print(f"\n📝 Test case: Pneumonia patient")
    print(f"   Text: {test_text[:100]}...")

    # Tokenize
    encoding = tokenizer(
        test_text,
        padding='max_length',
        truncation=True,
        max_length=384,
        return_tensors='pt'
    ).to(device)

    # Get prediction with attention
    with torch.no_grad():
        outputs = model(
            encoding['input_ids'],
            encoding['attention_mask'],
            concept_embeddings,
            return_attention=True
        )

        # Get diagnosis prediction
        diagnosis_logits = outputs['logits']
        diagnosis_probs = torch.sigmoid(diagnosis_logits).cpu().numpy()[0]
        predicted_dx = TARGET_CODES[diagnosis_probs.argmax()]
        confidence = diagnosis_probs.max()

        # Get concept scores
        concept_scores = torch.sigmoid(outputs['concept_scores']).cpu().numpy()[0]

        # Get top concepts
        top_indices = np.argsort(concept_scores)[::-1][:10]

        concepts = []
        for idx in top_indices:
            cui = concept_store['idx_to_concept'].get(idx, f'CUI_{idx}')
            concept_info = concept_store['concepts'].get(cui, {})

            concepts.append({
                'idx': idx,
                'cui': cui,
                'name': concept_info.get('preferred_name', f'Concept_{idx}'),
                'score': float(concept_scores[idx]),
                'semantic_types': concept_info.get('semantic_types', [])
            })

    print(f"\n  🎯 Predicted diagnosis: {predicted_dx} ({confidence:.1%} confidence)")
    print(f"  📊 Top concepts: {concepts[0]['name']}, {concepts[1]['name']}, {concepts[2]['name']}")

    # Extract evidence
    print(f"\n  🔍 Extracting evidence spans...")

    evidence = extract_evidence_spans(
        text=test_text,
        input_ids=encoding['input_ids'][0],
        attention_weights=outputs['attention_weights'],
        concepts=concepts,
        tokenizer=tokenizer,
        top_k=5
    )

    print(f"\n  ✅ Extracted {len(evidence)} evidence chains")

    # Show sample evidence
    if evidence:
        sample = evidence[0]
        print(f"\n  📋 Sample evidence chain:")
        print(f"     Concept: {sample['concept']}")
        print(f"     Score: {sample['score']:.1%}")
        print(f"     Evidence spans:")
        for i, span in enumerate(sample['evidence_spans'], 1):
            print(f"       {i}. \"{span}\"")

    # Save 043 checkpoint
    print(f"\n💾 Saving 043 checkpoint...")

    save_data = {
        'model_state_dict': model.state_dict(),
        'num_concepts': num_concepts,
        'concept_cuis': concept_cuis,
        'concept_names': concept_names,
        'concept_embeddings': concept_embeddings,
        'target_codes': TARGET_CODES,
        'phase': 'Phase 2 - Evidence Extraction + Knowledge Retrieval',
        'base_checkpoint': str(CHECKPOINT_042),
        'modifications': [
            'Added attention weight capture in EnhancedCrossAttention',
            'Added return_attention parameter to ShifaMindModel043',
            'Added extract_evidence_spans function',
            'No architectural changes - code-only modification'
        ],
        'test_results': {
            'test_case': 'Pneumonia patient',
            'predicted_diagnosis': predicted_dx,
            'confidence': float(confidence),
            'num_evidence_chains': len(evidence),
            'sample_evidence': evidence[0] if evidence else None
        }
    }

    torch.save(save_data, CHECKPOINT_043)

    print(f"  ✅ Saved: {CHECKPOINT_043}")
    print(f"  Size: {CHECKPOINT_043.stat().st_size / (1024**2):.1f} MB")

    # Summary
    print(f"\n" + "="*80)
    print("PHASE 2 MODEL CREATION COMPLETE")
    print("="*80)

    print(f"\n✅ ShifaMind043 checkpoint created successfully!")
    print(f"\n📊 Key features added:")
    print(f"   ✓ Attention weight capture from layers 9 and 11")
    print(f"   ✓ Evidence span extraction (avg {len(evidence[0]['evidence_spans'])} spans per concept)")
    print(f"   ✓ Compatible with 042 training (no retraining needed)")
    print(f"   ✓ Ready for knowledge retrieval integration")

    print(f"\n📁 Next steps:")
    print(f"   1. Run generate_knowledge_base_043.py to create knowledge base")
    print(f"   2. Run 043_eval.py for comprehensive Phase 2 evaluation")
    print(f"   3. Run 043_demo.py for interactive demo with evidence")

    print("="*80)


if __name__ == '__main__':
    main()
