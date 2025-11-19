# ShifaMind 041 Integration Guide

## How to Complete and Deploy 041.py

This guide explains how to complete the 041.py script by integrating it with components from 040.py.

## Current Status

### ✅ Completed Components in 041.py

1. **Imports and Configuration** (Lines 1-100)
   - All necessary imports
   - Path configuration for Google Drive
   - Settings for demo mode and caching

2. **Data Validation** (Lines 101-150)
   - `validate_data_structure()` function
   - Checks all required files exist

3. **Optimized Data Loading** (Lines 151-400)
   - `OptimizedDataLoader` class with cache support
   - `FastUMLSLoader` class for UMLS loading
   - `MIMICLoader` class for MIMIC-IV data

4. **Output Management** (Lines 401-500)
   - `OutputManager` class for organized file saving
   - Metrics, figures, and model checkpoint management

5. **Four Explainability Fixes** (Lines 501-750)
   - ✅ `ConceptPostProcessor` (Fix 1A)
   - ✅ `CitationMetrics` (Fix 2A)
   - ✅ `AlignmentEvaluator` (Fix 3A)
   - ✅ `ReasoningChainGenerator` (Fix 4A)

6. **Data Loading Execution** (Lines 751-900)
   - UMLS loading with cache
   - MIMIC-IV data loading
   - Dataset preparation and splitting

### 📋 Components Needed from 040.py

To complete 041.py, you need to add these sections from 040.py:

1. **Concept Store Building** (040.py lines 728-747)
   - Already included in `041_continuation.py`
   - ConceptStore class
   - Build concept set
   - PMI labeler

2. **Model Architecture** (040.py lines 854-1023)
   - Already included in `041_continuation.py`
   - EnhancedCrossAttention
   - ShifaMindModel
   - ClinicalDataset

3. **Training Stages** (040.py lines 1024-1454)
   - Stage 1: Diagnosis head (3 epochs)
   - Stage 2: Concept labels generation
   - Stage 3: Concept head (2 epochs)
   - Stage 4: Joint training (5 epochs)

4. **Evaluation with Fixes** (040.py lines 1577-1703)
   - Enhanced with 4 fixes
   - Citation metrics computation
   - Alignment score computation
   - Reasoning chain generation

## Step-by-Step Integration

### Option 1: Quick Integration (Recommended)

Simply append the content from `041_continuation.py` to the end of `041.py` before the final print statements:

```bash
# In Google Colab or locally:
cat 041_continuation.py >> 041.py
```

Then add the training and evaluation sections from 040.py.

### Option 2: Manual Integration

#### Step 1: Add Concept Store Components (after line 893 in 041.py)

```python
# ============================================================================
# BUILD CONCEPT STORE
# ============================================================================

print("\n" + "="*70)
print("CONCEPT STORE BUILDING")
print("="*70)

# Initialize tokenizer and base model
print("\nInitializing Bio_ClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
base_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

# Build concept store (use classes from 041_continuation.py)
concept_store = ConceptStore(umls_concepts, umls_loader_instance.icd10_to_cui)
concept_store.build_concept_set(TARGET_CODES, icd_descriptions, target_concept_count=150)

# Create concept embeddings
concept_embeddings = concept_store.create_concept_embeddings(tokenizer, base_model, device)

# Build PMI labeler
diagnosis_labeler = DiagnosisConditionalLabeler(
    concept_store,
    umls_loader_instance.icd10_to_cui,
    pmi_threshold=1.0
)
concepts_with_pmi = diagnosis_labeler.build_cooccurrence_statistics(df_train, TARGET_CODES)

# Filter concepts (from 040.py lines 753-843)
# ... add PMI filtering logic ...

print("✅ Concept store building complete")
```

#### Step 2: Add Model Initialization (after concept store)

```python
# ============================================================================
# INITIALIZE MODEL
# ============================================================================

print("\n" + "="*70)
print("MODEL INITIALIZATION")
print("="*70)

# Initialize ShifaMind model (use class from 041_continuation.py)
shifamind_model = ShifaMindModel(
    base_model=base_model,
    num_concepts=len(concept_store.concepts),
    num_classes=len(TARGET_CODES),
    fusion_layers=[9, 11]
).to(device)

print(f"Model parameters: {sum(p.numel() for p in shifamind_model.parameters()):,}")
print("✅ Model initialized")
```

#### Step 3: Add Training Stages (from 040.py)

Copy training stages 1-4 from 040.py lines 1024-1454, but use the new checkpoint paths:

```python
# Stage 1: Diagnosis Head
if not CHECKPOINT_DIAGNOSIS.exists():
    # ... training code from 040.py ...
    torch.save(checkpoint, CHECKPOINT_DIAGNOSIS)

# Stage 2: Generate Concept Labels
train_concept_labels = diagnosis_labeler.generate_dataset_labels(
    df_train,
    cache_file=str(OUTPUT_PATH / 'concept_labels_train.pkl')
)
# ... same for val and test ...

# Stage 3: Concept Head
if not CHECKPOINT_CONCEPTS.exists():
    # ... training code from 040.py ...
    torch.save(checkpoint, CHECKPOINT_CONCEPTS)

# Stage 4: Joint Training
if not CHECKPOINT_FINAL.exists():
    # ... training code from 040.py ...
    torch.save(checkpoint, CHECKPOINT_FINAL)
```

#### Step 4: Add Enhanced Evaluation with Fixes

```python
# ============================================================================
# EVALUATION WITH EXPLAINABILITY FIXES
# ============================================================================

print("\n" + "="*70)
print("EVALUATION WITH 4 EXPLAINABILITY FIXES")
print("="*70)

# Load best model
checkpoint = torch.load(CHECKPOINT_FINAL, map_location=device)
shifamind_model.load_state_dict(checkpoint['model_state_dict'])
shifamind_model.eval()

# Initialize fix components
concept_processor = ConceptPostProcessor(concept_store, DIAGNOSIS_KEYWORDS)
citation_metrics = CitationMetrics(min_concepts_threshold=3)
alignment_evaluator = AlignmentEvaluator()
reasoning_generator = ReasoningChainGenerator(ICD_DESCRIPTIONS)

# Run evaluation on test set
all_predictions = []
all_filtered_concepts = []
all_diagnosis_codes = []
all_labels = []

test_dataset = ClinicalDataset(
    df_test['text'].tolist(),
    df_test['labels'].tolist(),
    tokenizer,
    concept_labels=test_concept_labels
)
test_loader = DataLoader(test_dataset, batch_size=16)

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = shifamind_model(input_ids, attention_mask, concept_embeddings)

        # Get diagnosis predictions
        diagnosis_probs = torch.sigmoid(outputs['logits']).cpu().numpy()
        concept_scores_batch = torch.sigmoid(outputs['concept_scores']).cpu().numpy()

        # For each sample in batch
        for i in range(len(diagnosis_probs)):
            # Get predicted diagnosis
            pred_idx = np.argmax(diagnosis_probs[i])
            diagnosis_code = TARGET_CODES[pred_idx]
            diagnosis_conf = diagnosis_probs[i][pred_idx]

            # FIX 1A: Apply aggressive post-processing filter
            filtered_concepts = concept_processor.filter_concepts(
                concept_scores=concept_scores_batch[i],
                diagnosis_code=diagnosis_code,
                threshold=0.7,
                max_concepts=5
            )

            all_predictions.append(diagnosis_probs[i])
            all_filtered_concepts.append(filtered_concepts)
            all_diagnosis_codes.append(diagnosis_code)
            all_labels.append(labels[i].cpu().numpy())

# Convert to arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
pred_binary = (all_predictions > 0.5).astype(int)

# Compute diagnostic metrics
from sklearn.metrics import f1_score, precision_score, recall_score
diagnostic_f1 = f1_score(all_labels, pred_binary, average='macro', zero_division=0)
diagnostic_precision = precision_score(all_labels, pred_binary, average='macro', zero_division=0)
diagnostic_recall = recall_score(all_labels, pred_binary, average='macro', zero_division=0)

print(f"\n📊 Diagnostic Performance:")
print(f"   F1: {diagnostic_f1:.4f}")
print(f"   Precision: {diagnostic_precision:.4f}")
print(f"   Recall: {diagnostic_recall:.4f}")

# FIX 2A: Compute citation completeness metrics
citation_results = citation_metrics.compute_metrics(
    predicted_concepts=all_filtered_concepts,
    diagnosis_predictions=all_diagnosis_codes
)

print(f"\n📊 Citation Metrics:")
print(f"   Completeness: {citation_results['citation_completeness']:.1%}")
print(f"   Avg Concepts/Sample: {citation_results['avg_concepts_per_sample']:.2f}")
print(f"   Samples with ≥{citation_results['min_threshold']} concepts: {citation_results['samples_with_min_concepts']}/{citation_results['total_samples']}")

# FIX 3A: Compute alignment scores
# (Need ground truth concepts for this - extract from test_concept_labels)
ground_truth_concepts = []
for i, labels in enumerate(test_concept_labels):
    gt_cuis = [
        list(concept_store.concepts.keys())[j]
        for j, label in enumerate(labels)
        if label == 1
    ]
    ground_truth_concepts.append(gt_cuis)

alignment_results = alignment_evaluator.compute_alignment(
    predicted_concepts=all_filtered_concepts,
    ground_truth_concepts=ground_truth_concepts
)

print(f"\n📊 Alignment Metrics:")
print(f"   Overall Alignment: {alignment_results['overall_alignment']:.1%}")
print(f"   Std Alignment: {alignment_results['std_alignment']:.3f}")
print(f"   Min/Max: {alignment_results['min_alignment']:.3f} / {alignment_results['max_alignment']:.3f}")

# FIX 4A: Generate reasoning chains (sample)
print(f"\n📝 Sample Reasoning Chains:")
for i in range(min(3, len(all_filtered_concepts))):
    reasoning_chain = reasoning_generator.generate_chain(
        diagnosis_code=all_diagnosis_codes[i],
        diagnosis_confidence=all_predictions[i][TARGET_CODES.index(all_diagnosis_codes[i])],
        concepts=all_filtered_concepts[i],
        evidence_spans=None  # Would extract from attention if needed
    )
    print(f"\n--- Sample {i+1} ---")
    print(reasoning_chain)
    print()

# Save all results
final_results = {
    'diagnostic_f1': float(diagnostic_f1),
    'diagnostic_precision': float(diagnostic_precision),
    'diagnostic_recall': float(diagnostic_recall),
    'citation_metrics': citation_results,
    'alignment_metrics': alignment_results
}

output_manager.save_metrics(final_results, 'complete_evaluation_results.json')
output_manager.save_metrics(citation_results, 'citation_metrics.json')
output_manager.save_metrics(alignment_results, 'alignment_metrics.json')
output_manager.generate_report(final_results)

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE")
print("="*70)
print(f"   Diagnostic F1: {diagnostic_f1:.4f}")
print(f"   Citation Completeness: {citation_results['citation_completeness']:.1%}")
print(f"   Alignment Score: {alignment_results['overall_alignment']:.1%}")
print(f"\n📊 Full report: {OUTPUT_PATH / 'RESULTS_REPORT.md'}")
```

## Testing the Integration

### 1. Syntax Check

```bash
python -m py_compile 041.py
```

### 2. Quick Test (Demo Mode)

```python
# Set in 041.py:
DEMO_MODE = True
USE_CACHE = True

# Run:
python 041.py
```

Expected output:
- Data validation passes
- UMLS loads from cache (<1 min)
- Dataset prepared (smaller size)
- Model trains through 4 stages
- Evaluation shows all 4 fix metrics

### 3. Full Run

```python
# Set in 041.py:
DEMO_MODE = False
USE_CACHE = True

# Run:
python 041.py
```

## Expected Output Structure

After successful run, you should have:

```
04_Results/experiments/041_explainability_fixes/
├── metrics/
│   ├── citation_metrics.json
│   ├── alignment_metrics.json
│   └── complete_evaluation_results.json
├── figures/
│   └── (add visualization code as needed)
├── models/
│   └── best_model.pt
└── RESULTS_REPORT.md

03_Models/checkpoints/
├── shifamind_041_diagnosis.pt
├── shifamind_041_concepts.pt
└── shifamind_041_final.pt
```

## Verification Checklist

After integration and running:

- [ ] Script runs without errors
- [ ] UMLS cache loads successfully
- [ ] All 4 training stages complete
- [ ] Checkpoints saved correctly
- [ ] Citation completeness >95%
- [ ] Alignment score >70%
- [ ] Diagnostic F1 ≥0.76
- [ ] RESULTS_REPORT.md generated
- [ ] All metrics JSON files created

## Common Issues and Solutions

### Issue: "ConceptStore not defined"
**Solution**: Ensure classes from `041_continuation.py` are added before use

### Issue: "Checkpoint already exists"
**Solution**: Script skips training if checkpoints exist. Delete checkpoints to retrain.

### Issue: "Out of memory"
**Solution**:
- Reduce batch size
- Enable DEMO_MODE
- Use gradient checkpointing

### Issue: "Low citation completeness (<95%)"
**Solution**:
- Lower threshold in `ConceptPostProcessor` from 0.7 to 0.6
- Increase max_concepts from 5 to 7

### Issue: "Low alignment score (<70%)"
**Solution**:
- Check PMI threshold (try lowering from 1.0 to 0.8)
- Verify keyword lists in DIAGNOSIS_KEYWORDS

## Next Steps

1. Complete integration following this guide
2. Run full training pipeline
3. Analyze metrics in `04_Results/experiments/041_explainability_fixes/`
4. Iterate on thresholds if needed
5. Deploy to production

## Support

For integration issues:
- Check Python syntax with `py_compile`
- Review error messages carefully
- Compare with 040.py structure
- Ensure all imports are present

---

**Version**: 041
**Date**: November 2025
