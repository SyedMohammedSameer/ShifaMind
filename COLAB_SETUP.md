# Google Colab Setup Guide

This guide explains how to run ShifaMind in Google Colab.

## Quick Setup

### Option 1: Clone from GitHub (Recommended)

```python
# Clone the repository
!git clone https://github.com/SyedMohammedSameer/ShifaMind.git
%cd ShifaMind

# Install only what's needed (Colab already has most packages)
!pip install -q -r requirements_colab.txt

# Mount Google Drive (for MIMIC-IV data)
from google.colab import drive
drive.mount('/content/drive')
```

### Option 2: Upload Files Manually

If you prefer to upload files directly:

1. Upload all `.py` files to Colab:
   - `config.py`
   - `final_concept_filter.py`
   - `final_knowledge_base_generator.py`
   - `final_inference.py`
   - `final_model_training.py`
   - `final_evaluation.py`
   - `final_demo.py`

2. Install only what's missing in Colab:
```python
!pip install -q gradio
# That's it! Colab already has torch, transformers, scikit-learn, pandas, etc.
```

---

## Running the Pipeline

### Step 1: Configure Paths

Edit `config.py` to point to your Google Drive:

```python
# In config.py, update BASE_PATH:
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
```

### Step 2: Generate Knowledge Base

```python
# Run knowledge base generator
!python final_knowledge_base_generator.py
```

**Expected output:**
```
🏥 ShifaMind Clinical Knowledge Base Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📂 Loading UMLS Concepts from: /content/drive/MyDrive/ShifaMind/00_Data/UMLS/...
  ✅ Loaded 51,420 unique concepts
...
```

### Step 3: Train Model (Long-running)

```python
# Run training (takes several hours on GPU)
!python final_model_training.py
```

**Note:** Training requires significant GPU time. Consider using:
- Colab Pro for longer GPU sessions
- Checkpointing to resume training

**Expected stages:**
1. Stage 1: Diagnosis head training (3 epochs)
2. Stage 2: Concept head training (2 epochs)
3. Stage 3: Joint fine-tuning (3 epochs)

### Step 4: Run Evaluation

```python
# Evaluate the trained model
!python final_evaluation.py
```

### Step 5: Launch Demo

```python
# Launch interactive Gradio demo
!python final_demo.py
```

The demo will provide a public URL you can share.

---

## Python API Usage (Recommended for Colab)

Instead of using shell commands (`!python`), you can import and use the modules directly:

### Quick Inference Example

```python
from final_inference import ShifaMindPredictor

# Initialize predictor
predictor = ShifaMindPredictor()

# Predict on clinical note
clinical_note = """
72-year-old male presents with fever (38.9°C), productive cough,
and shortness of breath. Chest X-ray shows right lower lobe infiltrate.
"""

result = predictor.predict(clinical_note)

# Display results
print(f"Diagnosis: {result['diagnosis']['name']}")
print(f"Code: {result['diagnosis']['code']}")
print(f"Confidence: {result['diagnosis']['confidence']:.1%}")

print("\nTop Clinical Concepts:")
for concept in result['concepts'][:5]:
    print(f"  - {concept['name']} ({concept['score']:.1%})")
```

### Training from Python

```python
import sys
sys.path.append('/content/ShifaMind')

from final_model_training import main
from pathlib import Path

# Create args object
class Args:
    output_path = Path('/content/drive/MyDrive/ShifaMind/03_Models/training_results')
    max_samples_per_code = 20000
    retrain = False

args = Args()
args.output_path.mkdir(parents=True, exist_ok=True)

# Run training
main(args)
```

---

## Common Issues & Solutions

### Issue 1: Import Errors

**Error:** `ModuleNotFoundError: No module named 'config'`

**Solution:** Ensure you're in the correct directory:
```python
import os
print(os.getcwd())  # Should show /content/ShifaMind

# If not, change directory:
%cd /content/ShifaMind
```

### Issue 2: CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:** Use CPU or reduce batch size:
```python
# Option 1: Use CPU
predictor = ShifaMindPredictor(device='cpu')

# Option 2: Edit config.py and reduce BATCH_SIZE
# Change from 8 to 4 or 2
```

### Issue 3: Files Not Found

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: '/content/drive/MyDrive/ShifaMind/00_Data/...'`

**Solution:**
1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Verify path exists:
```python
!ls /content/drive/MyDrive/ShifaMind/
```

3. Update `config.py` with correct paths

### Issue 4: Argparse Errors (FIXED)

**Previous Error:** `error: unrecognized arguments: -f /root/.local/share/jupyter/runtime/kernel-...`

**Status:** ✅ Fixed in latest version using `parse_known_args()`

If you still see this error, make sure you've pulled the latest code from GitHub.

---

## Data Requirements

### Required Data Files

Ensure these files exist in your Google Drive:

```
/content/drive/MyDrive/ShifaMind/
├── 00_Data/
│   ├── UMLS/
│   │   ├── MRCONSO.RRF
│   │   ├── MRSTY.RRF
│   │   └── icd10cm_codes_2023.txt
│   └── MIMIC-IV/
│       ├── noteevents.csv.gz
│       ├── diagnoses_icd.csv.gz
│       ├── patients.csv.gz
│       └── admissions.csv.gz
├── 01_Processed/
│   └── (will be created during processing)
├── 02_Checkpoints/
│   └── (will be created during training)
└── 03_Models/
    └── (will be created for final models)
```

### Obtaining MIMIC-IV Data

1. Complete CITI training: https://physionet.org/about/citi-course/
2. Request access: https://physionet.org/content/mimiciv/
3. Download required files
4. Upload to Google Drive

### Obtaining UMLS Data

1. Register for UMLS account: https://www.nlm.nih.gov/research/umls/
2. Download Metathesaurus files
3. Extract MRCONSO.RRF and MRSTY.RRF
4. Upload to Google Drive

---

## Performance Tips

### 1. Use GPU Runtime

**Settings → Runtime type → Hardware accelerator → GPU**

### 2. Enable High-RAM

**Settings → Runtime type → Runtime shape → High-RAM**

### 3. Keep Session Alive

Colab sessions timeout after inactivity. Use this script:

```javascript
function KeepAlive() {
    console.log("Keeping session alive");
    document.querySelector("colab-connect-button").click();
}
setInterval(KeepAlive, 60000);  // Click every 60 seconds
```

Paste in browser console (F12).

### 4. Save Checkpoints Frequently

Training automatically saves checkpoints after each stage to Google Drive.

---

## Complete Colab Notebook Example

Here's a complete notebook you can copy-paste:

```python
# ============================================================================
# SHIFAMIND - Complete Colab Setup
# ============================================================================

# 1. Setup environment
!git clone https://github.com/SyedMohammedSameer/ShifaMind.git
%cd ShifaMind
!pip install -q gradio  # Only install what's missing

# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 4. Generate knowledge base (5-10 minutes)
print("\n🏥 Generating knowledge base...")
!python final_knowledge_base_generator.py

# 5. Train model (several hours)
print("\n🎓 Training model...")
!python final_model_training.py

# 6. Run evaluation
print("\n📊 Evaluating model...")
!python final_evaluation.py

# 7. Test inference
print("\n🔬 Testing inference...")
from final_inference import ShifaMindPredictor

predictor = ShifaMindPredictor()
result = predictor.predict("""
72-year-old male with fever, cough, and chest infiltrate.
""")

print(f"\nDiagnosis: {result['diagnosis']['name']}")
print(f"Confidence: {result['diagnosis']['confidence']:.1%}")

# 8. Launch demo
print("\n🚀 Launching demo...")
!python final_demo.py
```

---

## Next Steps

- See [README.md](README.md) for project overview
- See [docs/USAGE.md](docs/USAGE.md) for detailed API documentation
- See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for development status

---

**For questions or issues, please open a GitHub issue.**

**Author:** Mohammed Sameer Syed
**Institution:** University of Arizona
**Date:** November 20, 2025
