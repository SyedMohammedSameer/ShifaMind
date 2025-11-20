# Quick Start Guide for Running 041.py in Google Colab

## Problem

Getting this error when trying to run `041.py` in Google Colab?

```
AttributeError: partially initialized module 'torch' has no attribute 'autograd'
(most likely due to a circular import)
```

## Solution (3 Simple Steps)

### Step 1: Fix PyTorch

In a Colab cell, run:

```python
!python COLAB_FIX.py
```

This will automatically fix the PyTorch circular import error. Follow the prompts.

### Step 2: Restart Runtime

**IMPORTANT:** After the fix completes:
- Go to: **Runtime → Restart runtime**
- This ensures the fixes take effect

### Step 3: Run Your Script

In a new Colab cell, run:

```python
%run 041.py
```

**That's it!** Your script should now run without errors.

---

## Alternative: Quick Manual Fix

If you prefer to fix it manually:

### Option A: Reinstall PyTorch

```python
# Cell 1: Reinstall
!pip install --upgrade --force-reinstall torch torchvision torchaudio

# Then: Runtime → Restart runtime

# Cell 2: Run script
%run 041.py
```

### Option B: Use the Runner Script

```python
!python run_041_colab.py
```

This checks your environment and runs the script safely.

---

## Complete Setup from Scratch

If you're setting up for the first time in Colab:

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Navigate to project
%cd /content/drive/MyDrive/ShifaMind

# Cell 3: Fix PyTorch
!python COLAB_FIX.py

# Then: Runtime → Restart runtime

# Cell 4: Install dependencies
!pip install transformers scikit-learn pandas tqdm matplotlib

# Cell 5: Run the script
%run 041.py
```

---

## Files Provided

- **`COLAB_FIX.py`** - Automatic fix for PyTorch errors
- **`run_041_colab.py`** - Safe runner with environment checks
- **`COLAB_TORCH_FIX.md`** - Detailed troubleshooting guide
- **`041.py`** - Your main training script

---

## Common Issues

### Issue: "Python 3.12 compatibility warning"

**Solution:** Python 3.12 may have issues with PyTorch. Options:
1. Use Google Colab's default Python (usually 3.10)
2. Run locally with Python 3.10 or 3.11

### Issue: "CUDA not available"

**Solution:** Enable GPU in Colab:
1. Go to: **Runtime → Change runtime type**
2. Select: **Hardware accelerator → GPU** (T4 or better)
3. Click **Save**

### Issue: Script runs but very slow

**Solution:**
1. Make sure GPU is enabled (see above)
2. Check `device` output - should say `cuda` not `cpu`

### Issue: Out of memory errors

**Solution:**
1. Use a Colab Pro account for more RAM
2. Or reduce batch size in the script (line ~1270, change `batch_size=8` to `batch_size=4`)

---

## Still Having Problems?

Read the detailed troubleshooting guide: **`COLAB_TORCH_FIX.md`**

Or run the diagnostic:

```python
!python run_041_colab.py
```

This will check your environment and tell you exactly what's wrong.

---

## Expected Runtime

Running `041.py` in Colab will take:
- **Stage 1 (Diagnosis):** ~1-2 hours
- **Stage 2 (Concepts):** ~2-3 hours
- **Stage 3 (Joint):** ~2-3 hours
- **Total:** ~5-8 hours depending on GPU

**Tip:** Keep the Colab tab open to prevent disconnection.

---

## Need Help?

1. Check `COLAB_TORCH_FIX.md` for detailed troubleshooting
2. Run `!python run_041_colab.py` for environment diagnostics
3. Make sure you've restarted the runtime after running the fix

**Good luck with your training!** 🚀
