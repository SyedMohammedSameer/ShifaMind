# ShifaMind - Dead Simple Colab Setup

**NO Google Drive needed. NO path configuration needed. Just 3 simple steps.**

---

## Step 1: Clone the Repo

```python
!git clone https://github.com/SyedMohammedSameer/ShifaMind_Capstone.git
%cd ShifaMind_Capstone

# Install Gradio (only thing Colab doesn't have)
!pip install -q gradio
```

---

## Step 2: Run Setup Script

```python
!python setup_colab.py
```

This creates the folder structure:
```
/content/ShifaMind_Data/
├── UMLS/          ← Upload your UMLS files here
├── ICD10/         ← Upload your ICD-10 files here
├── MIMIC/         ← Upload your MIMIC files here
├── Models/        ← Outputs will go here
└── Results/       ← Results will go here
```

---

## Step 3: Upload Your Data Files

### Option A: Using Colab's File Browser (EASIEST)

1. Click the 📁 folder icon on the left sidebar
2. Navigate to `/content/ShifaMind_Data/UMLS/`
3. Click the upload button (looks like a page with an arrow)
4. Upload these files:
   - `MRCONSO.RRF`
   - `MRSTY.RRF`

5. Navigate to `/content/ShifaMind_Data/ICD10/`
6. Upload:
   - `icd10cm-codes-2024.txt`

7. Navigate to `/content/ShifaMind_Data/MIMIC/`
8. Upload (for training):
   - `noteevents.csv.gz`
   - `diagnoses_icd.csv.gz`
   - `patients.csv.gz`
   - `admissions.csv.gz`

### Option B: Using Python Upload Widget

```python
from google.colab import files
import shutil

# Upload UMLS files
print("📤 Upload MRCONSO.RRF:")
uploaded = files.upload()
for filename in uploaded.keys():
    shutil.move(filename, f'/content/ShifaMind_Data/UMLS/{filename}')

print("📤 Upload MRSTY.RRF:")
uploaded = files.upload()
for filename in uploaded.keys():
    shutil.move(filename, f'/content/ShifaMind_Data/UMLS/{filename}')

# Upload ICD-10 file
print("📤 Upload icd10cm-codes-2024.txt:")
uploaded = files.upload()
for filename in uploaded.keys():
    shutil.move(filename, f'/content/ShifaMind_Data/ICD10/{filename}')

# Upload MIMIC files (repeat for each file)
print("📤 Upload noteevents.csv.gz:")
uploaded = files.upload()
for filename in uploaded.keys():
    shutil.move(filename, f'/content/ShifaMind_Data/MIMIC/{filename}')
```

---

## Step 4: Verify Files Are Uploaded

```python
!python setup_colab.py
```

You should see ✅ checkmarks for all files.

---

## Step 5: Run the Pipeline

### Generate Knowledge Base (5-10 min)

```python
!python final_knowledge_base_generator.py
```

### Train Model (Several Hours - GPU Recommended)

```python
!python final_model_training.py
```

### Evaluate Model

```python
!python final_evaluation.py
```

### Launch Demo

```python
!python final_demo.py
```

---

## Complete Copy-Paste Script

Here's everything in one block:

```python
# ============================================================================
# SHIFAMIND - COMPLETE COLAB SETUP (COPY-PASTE THIS)
# ============================================================================

# 1. Clone repo and install dependencies
!git clone https://github.com/SyedMohammedSameer/ShifaMind_Capstone.git
%cd ShifaMind_Capstone
!pip install -q gradio

# 2. Setup folder structure
!python setup_colab.py

# 3. NOW UPLOAD YOUR FILES (see instructions above)
print("\n⚠️  STOP HERE AND UPLOAD YOUR DATA FILES ⚠️")
print("Upload files to /content/ShifaMind_Data/ folders")
print("Then run the rest of the code below...")

# 4. Verify files (run after uploading)
!python setup_colab.py

# 5. Generate knowledge base
!python final_knowledge_base_generator.py

# 6. Train model (takes several hours)
!python final_model_training.py

# 7. Evaluate
!python final_evaluation.py

# 8. Launch demo
!python final_demo.py
```

---

## FAQ

### Q: Do I need Google Drive?
**A:** No! Everything runs in Colab's local storage (`/content/`). Just upload files directly.

### Q: Where do I get UMLS files?
**A:**
1. Register at https://www.nlm.nih.gov/research/umls/
2. Download Metathesaurus
3. Extract `MRCONSO.RRF` and `MRSTY.RRF`

### Q: Where do I get ICD-10 codes?
**A:** Download from https://www.cms.gov/medicare/coding-billing/icd-10-codes

### Q: Where do I get MIMIC-IV?
**A:**
1. Complete CITI training: https://physionet.org/about/citi-course/
2. Request access: https://physionet.org/content/mimiciv/

### Q: What if Colab disconnects during training?
**A:** Training automatically saves checkpoints. If disconnected, re-run and it will resume from the last checkpoint.

### Q: Can I use CPU instead of GPU?
**A:** Yes, but training will be VERY slow (days instead of hours). For testing, you can skip training and use pre-trained models.

### Q: How much storage do I need?
**A:**
- UMLS files: ~5 GB
- MIMIC-IV: ~20 GB
- Models: ~2 GB
- **Total: ~30 GB** (Colab free tier has ~100 GB)

---

## Troubleshooting

### Error: "FileNotFoundError: MRCONSO.RRF"

**Fix:** Upload `MRCONSO.RRF` to `/content/ShifaMind_Data/UMLS/`

### Error: "ImportError: No module named 'gradio'"

**Fix:** Run `!pip install gradio`

### Error: "CUDA out of memory"

**Fix:** The model is too large for Colab's free GPU. Options:
1. Use Colab Pro for more GPU memory
2. Run on CPU (very slow): Edit `config.py` and force CPU mode

---

**That's it! No path configuration, no Google Drive setup, just upload and run.** 🎉

---

**Author:** Mohammed Sameer Syed
**Institution:** University of Arizona
**Date:** November 20, 2025
