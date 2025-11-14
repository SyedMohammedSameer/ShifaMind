# ShifaMind Phase 4 Proposal: Precision Concept Selection + Smart RAG

## Executive Summary

**Goal**: Achieve 0.80+ F1 with high-quality concept selection (70%+ precision)

**Status**: Phase 3 achieved 0.7734 F1 but with poor concept quality (24.6 concepts, ~30% precision)

**Strategy**: Reset to Phase 2 baseline, add targeted improvements systematically

---

## 🔍 Root Cause Analysis

### What Went Wrong in Phase 3:

| Issue | Symptom | Impact |
|-------|---------|--------|
| **No Concept Supervision** | Activates 24.6 concepts (vs 13.2 in Phase 2) | ~30% precision, lots of noise |
| **Weak RAG Integration** | Only +0.0005 F1 improvement | RAG not helping predictions |
| **Loss Function Bug** | Negative loss (-0.0055) | Diversity term encourages MORE concepts |
| **Too Much Complexity** | 200 concepts, 3 layers, RAG | Hard to train without proper signals |

### Key Insight:
> **Without concept labels, the model has no idea which concepts are relevant.**
> It's like asking someone to pick ingredients for a recipe without telling them what dish they're making.

---

## 🎯 Phase 4: Three-Pillar Solution

### Pillar 1: ClinicalBERT Concept Pseudo-Labeling

**Problem**: No ground truth for which concepts should be active

**Solution**: Generate pseudo-labels using semantic similarity + keyword matching

```python
# For each training sample:
def generate_concept_labels(clinical_note, concept_list):
    labels = []

    # Step 1: Encode note with ClinicalBERT
    note_embedding = clinicalbert.encode(clinical_note)

    for concept in concept_list:
        score = 0

        # Strategy A: Semantic similarity
        concept_embedding = concept.embedding
        similarity = cosine_similarity(note_embedding, concept_embedding)
        score += similarity

        # Strategy B: Keyword matching (exact)
        if concept.name.lower() in clinical_note.lower():
            score += 0.3  # Boost

        # Strategy C: Synonym matching
        for synonym in concept.terms:
            if synonym.lower() in clinical_note.lower():
                score += 0.15

        # Threshold for pseudo-label
        label = 1 if score > 0.65 else 0
        labels.append(label)

    return labels
```

**Benefits**:
- ✅ Provides supervision for concept selection
- ✅ High precision (0.65 threshold = confident matches)
- ✅ Combines semantic + lexical matching
- ✅ Can generate labels for entire training set upfront

**Expected Impact**: Concepts 24.6 → ~8-12, Precision 30% → 70%+

---

### Pillar 2: Two-Stage Diagnosis-Aware RAG

**Problem**: RAG retrieves irrelevant documents (e.g., "Pancreatitis" for pneumonia)

**Solution**: Predict diagnosis first, then filter document retrieval

```python
# Two-pass forward during training
def forward_with_smart_rag(clinical_note):
    # STAGE 1: Initial prediction (no RAG)
    initial_output = model(clinical_note, rag_context=None)
    predicted_diagnosis = initial_output.logits.argmax()  # e.g., J189

    # STAGE 2: Diagnosis-aware RAG
    diagnosis_code = target_codes[predicted_diagnosis]  # "J189"

    # Filter documents by diagnosis category
    if diagnosis_code.startswith('J'):  # Respiratory
        allowed_docs = respiratory_documents
    elif diagnosis_code.startswith('A'):  # Infectious
        allowed_docs = infectious_documents
    # ... etc

    # STAGE 3: Retrieve from filtered set
    retrieved_docs = rag.retrieve(
        query=clinical_note,
        document_pool=allowed_docs,  # Filtered!
        k=5
    )

    # STAGE 4: Refined prediction with RAG
    final_output = model(
        clinical_note,
        rag_context=retrieved_docs
    )

    return final_output
```

**Benefits**:
- ✅ Retrieval precision: ~40% → 80%+
- ✅ No cross-contamination (lung docs for heart cases)
- ✅ Can increase RAG frequency to 50-60% (still fast)
- ✅ Cache filtered document sets (reusable)

**Expected Impact**: RAG becomes actually helpful, +0.03-0.05 F1

---

### Pillar 3: Staged Training Strategy

**Problem**: Training everything jointly from scratch is unstable

**Solution**: Train in stages with increasing complexity

```
Stage 1: Diagnosis Head (3 epochs)
├─ Freeze: Concept head, RAG
├─ Train: Base model + diagnosis classifier
└─ Goal: Strong diagnosis predictions (0.78+ F1)

Stage 2: Pseudo-Label Generation
├─ Use trained diagnosis head
├─ Generate concept labels for all training samples
├─ Save labels to disk
└─ Goal: High-quality concept supervision

Stage 3: Concept Head (2 epochs)
├─ Freeze: Diagnosis head
├─ Train: Concept head with pseudo-labels
└─ Goal: Selective concept activation (~10 concepts)

Stage 4: Joint Fine-Tuning with RAG (3 epochs)
├─ Train: Everything end-to-end
├─ RAG: 50% of batches, diagnosis-aware
└─ Goal: Final polish, 0.80+ F1
```

**Benefits**:
- ✅ Stable training (no conflicting signals)
- ✅ Each component learns its role
- ✅ Can validate each stage independently
- ✅ Better convergence than joint training

**Expected Impact**: More stable, higher final performance

---

## 📐 Proposed Phase 4 Architecture

### Model Configuration

```python
Phase4ShifaMind(
    # Core architecture
    concepts=150,              # Reduced from 200 (better quality)
    fusion_layers=[9, 11],     # 2 layers (vs 3 in Phase 3)
    attention_heads=8,
    sequence_length=384,

    # Concept selection
    concept_threshold=0.7,     # High confidence required
    pseudo_labeling=True,      # ClinicalBERT-based
    similarity_threshold=0.65,

    # RAG configuration
    rag_strategy="diagnosis_aware",  # Two-stage
    rag_frequency=0.5,         # 50% of batches
    rag_docs_per_query=5,

    # Training
    staged_training=True,
    stages=['diagnosis', 'pseudo_label', 'concepts', 'joint'],
)
```

### Loss Function (Stage-Specific)

```python
# Stage 1: Diagnosis only
loss_stage1 = BCEWithLogits(diagnosis_logits, labels)

# Stage 3: Concepts with pseudo-labels
loss_stage3 = (
    0.70 * BCEWithLogits(concept_logits, pseudo_labels) +
    0.30 * TopKConfidence(concept_probs, k=10)
)

# Stage 4: Joint
loss_stage4 = (
    0.50 * BCEWithLogits(diagnosis_logits, labels) +
    0.35 * BCEWithLogits(concept_logits, pseudo_labels) +
    0.15 * TopKConfidence(concept_probs, k=10)
)
```

---

## 📊 Expected Performance

| Metric | Phase 3 | Phase 4 Target | Improvement |
|--------|---------|----------------|-------------|
| **Macro F1** | 0.7734 | 0.80-0.83 | +0.027-0.057 |
| **Concepts Activated** | 24.6 | 8-12 | -14.6 (↓59%) |
| **Concept Precision** | ~30% | 70-80% | +40-50% |
| **RAG Retrieval Precision** | ~40% | 75-85% | +35-45% |
| **Training Time** | 17.8 min | 25-30 min | +7-12 min |

### Per-Class F1 Predictions

| Code | Phase 3 | Phase 4 Target | Reason |
|------|---------|----------------|--------|
| **J189** (Pneumonia) | 0.7044 | 0.78-0.80 | Better respiratory concepts |
| **I5023** (Heart Failure) | 0.8279 | 0.83-0.85 | Already good, minor gains |
| **A419** (Sepsis) | 0.7177 | 0.79-0.82 | Better infection concepts |
| **K8000** (Cholecystitis) | 0.8438 | 0.85-0.87 | Specific biliary concepts |

---

## 🔧 Implementation Plan

### Week 1: Foundation (Days 1-2)

**Day 1**: ClinicalBERT Concept Pseudo-Labeling
- [ ] Implement `ConceptPseudoLabeler` class
- [ ] Generate labels for all training samples
- [ ] Validate label quality (manual review of 50 samples)
- [ ] Save labels to disk

**Day 2**: Diagnosis-Aware RAG System
- [ ] Implement `DiagnosisAwareRAG` class
- [ ] Build diagnosis-to-document mappings
- [ ] Create filtered document stores
- [ ] Test retrieval precision

### Week 2: Staged Training (Days 3-5)

**Day 3**: Stage 1 - Diagnosis Head
- [ ] Train diagnosis head (3 epochs)
- [ ] Validate F1 ≥ 0.78
- [ ] Save checkpoint

**Day 4**: Stage 2 & 3 - Concept Head
- [ ] Generate pseudo-labels using trained diagnosis
- [ ] Train concept head (2 epochs)
- [ ] Validate concept precision ≥ 70%
- [ ] Save checkpoint

**Day 5**: Stage 4 - Joint Fine-Tuning
- [ ] Joint training with RAG (3 epochs)
- [ ] Monitor all metrics
- [ ] Final evaluation

### Week 3: Validation & Analysis (Days 6-7)

**Day 6**: Performance Analysis
- [ ] Test set evaluation
- [ ] Per-class F1 analysis
- [ ] Concept quality analysis
- [ ] RAG effectiveness analysis

**Day 7**: Demo & Documentation
- [ ] Create comprehensive demo
- [ ] Document findings
- [ ] Create visualizations
- [ ] Write final report

---

## 💻 Key Code Components

### 1. ConceptPseudoLabeler

```python
class ConceptPseudoLabeler:
    """Generate concept pseudo-labels using ClinicalBERT"""

    def __init__(self, tokenizer, model, concepts, threshold=0.65):
        self.tokenizer = tokenizer
        self.model = model
        self.concepts = concepts
        self.threshold = threshold

        # Pre-compute concept embeddings
        self.concept_embeddings = self._encode_concepts()

    def generate_labels(self, clinical_note: str) -> List[int]:
        """Generate binary labels for concepts"""
        # Encode note
        note_emb = self._encode_text(clinical_note)

        labels = []
        for cui, concept in self.concepts.items():
            score = 0

            # Semantic similarity
            concept_emb = self.concept_embeddings[cui]
            similarity = cosine_similarity(note_emb, concept_emb)
            score += similarity

            # Keyword exact match
            if self._exact_match(clinical_note, concept['name']):
                score += 0.3

            # Synonym matching
            for term in concept.get('terms', [])[:5]:
                if self._exact_match(clinical_note, term):
                    score += 0.15
                    break

            # Threshold
            label = 1 if score >= self.threshold else 0
            labels.append(label)

        return labels
```

### 2. DiagnosisAwareRAG

```python
class DiagnosisAwareRAG:
    """Two-stage RAG with diagnosis filtering"""

    def __init__(self, concept_store, document_store, target_codes):
        self.concept_store = concept_store
        self.document_store = document_store
        self.target_codes = target_codes

        # Build diagnosis-specific document pools
        self.diagnosis_doc_pools = self._build_filtered_pools()

    def retrieve_with_diagnosis(self, query_emb, predicted_diagnosis_idx):
        """Retrieve documents filtered by diagnosis"""
        # Get diagnosis code
        diagnosis_code = self.target_codes[predicted_diagnosis_idx]

        # Get filtered document pool
        doc_pool = self.diagnosis_doc_pools.get(
            diagnosis_code[0],  # First letter (J, I, A, K)
            self.document_store  # Fallback to all
        )

        # Retrieve from filtered pool
        retrieved = self._search(query_emb, doc_pool, k=5)

        return retrieved
```

### 3. StagedTrainer

```python
class StagedTrainer:
    """Multi-stage training orchestrator"""

    def train(self):
        # Stage 1: Diagnosis head
        print("Stage 1: Training diagnosis head...")
        self.train_diagnosis_head(epochs=3)

        # Stage 2: Generate pseudo-labels
        print("Stage 2: Generating concept pseudo-labels...")
        pseudo_labels = self.generate_pseudo_labels()

        # Stage 3: Concept head
        print("Stage 3: Training concept head...")
        self.train_concept_head(pseudo_labels, epochs=2)

        # Stage 4: Joint fine-tuning
        print("Stage 4: Joint fine-tuning with RAG...")
        self.joint_finetune(pseudo_labels, epochs=3)
```

---

## ⚠️ Risk Mitigation

### Risk 1: Pseudo-Labels Too Noisy
- **Mitigation**: High threshold (0.65), manual validation
- **Fallback**: Use only keyword matching (lower recall but high precision)

### Risk 2: Two-Stage RAG Too Slow
- **Mitigation**: Cache filtered doc pools, batch retrieval
- **Fallback**: Reduce RAG frequency to 30%

### Risk 3: Staged Training Doesn't Converge
- **Mitigation**: Save checkpoints after each stage, can restart
- **Fallback**: Skip to joint training with pseudo-labels

---

## 📈 Success Criteria

### Must-Have (Phase 4 Success)
- ✅ Macro F1 ≥ 0.80
- ✅ Concept precision ≥ 70%
- ✅ Avg concepts activated ≤ 12

### Nice-to-Have (Stretch Goals)
- 🎯 Macro F1 ≥ 0.82
- 🎯 All per-class F1 ≥ 0.75
- 🎯 Concept precision ≥ 80%

### Show-Stopper (Must Fix)
- ❌ F1 < 0.78 (worse than Phase 2)
- ❌ Training instability (NaN losses)
- ❌ Concept precision < 50%

---

## 🚀 Ready to Implement?

**Recommendation**: Start with **minimal viable Phase 4**:

1. ✅ ClinicalBERT pseudo-labeling (1-2 hours to implement)
2. ✅ Diagnosis-aware RAG (1 hour to implement)
3. ✅ Staged training (existing code + orchestration)
4. ✅ Test on small subset first (100 samples)
5. ✅ Full training if validation looks good

**Expected timeline**:
- Implementation: 4-6 hours
- Training: 25-30 minutes
- Evaluation: 30 minutes
- **Total**: 5-7 hours to Phase 4 results

---

## 📝 Questions Before Implementation

1. **Pseudo-label threshold**: Start with 0.65 or be more conservative (0.70)?
2. **RAG frequency**: 50% or start lower (30%)?
3. **Concept count**: 150 or reduce further to 120?
4. **Training stages**: All 4 stages or skip straight to joint?
5. **Validation set**: Use current split or create smaller dev set for faster iteration?

Let me know your preferences and I'll create the implementation! 🎯
