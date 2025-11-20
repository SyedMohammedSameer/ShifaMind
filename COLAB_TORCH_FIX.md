# Fix for PyTorch Circular Import Error in Colab

## The Error

```
AttributeError: partially initialized module 'torch' has no attribute 'autograd' (most likely due to a circular import)
```

This error occurs in `/usr/local/lib/python3.12/dist-packages/torch/jit/_builtins.py` when trying to import torch.

## Root Cause

This is a known issue in Google Colab caused by:
1. **Corrupted PyTorch installation** - Most common cause
2. **Python version incompatibility** - PyTorch may not be fully compatible with Python 3.12
3. **Stale import cache** - Old .pyc files interfering
4. **Runtime state issues** - Colab kernel needs refresh

## Solutions (Try in Order)

### Solution 1: Reinstall PyTorch (Recommended)

In a Colab cell, run:

```python
# Uninstall and reinstall PyTorch
!pip uninstall -y torch torchvision torchaudio
!pip install torch torchvision torchaudio

# Restart the runtime
# Go to: Runtime -> Restart runtime
```

After the restart, run your script again.

### Solution 2: Downgrade to Python 3.10

The error shows you're using Python 3.12, which may have compatibility issues:

```python
# This requires creating a new Colab notebook with Python 3.10
# Unfortunately, Colab doesn't easily support changing Python versions
# You may need to use a local environment instead
```

### Solution 3: Clear Import Cache

```python
import sys
import importlib

# Clear any cached imports
if 'torch' in sys.modules:
    del sys.modules['torch']

# Clear __pycache__
!find /content -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Now try importing
import torch
print(f"PyTorch {torch.__version__} loaded successfully!")
```

### Solution 4: Use the Fix Script (Easiest)

We've created a fix script that handles this automatically:

```python
# Upload COLAB_FIX.py to your Colab environment
# Then run:
!python COLAB_FIX.py
```

## Recommended Approach for Running 041.py in Colab

Instead of running the Python file directly, use this in a Colab cell:

```python
# Cell 1: Setup and fix imports
!pip install --upgrade --force-reinstall torch

# Cell 2: Mount Drive (if needed)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Install dependencies
!pip install transformers scikit-learn tqdm pandas

# Cell 4: Run the script
%run -i 041.py
```

## Alternative: Convert to Jupyter Notebook

Since 041.py is a long script, it may work better as a Jupyter notebook:

1. Upload `041.py` to Colab
2. Use `!p2j 041.py` to convert it to a notebook (requires jupytext)
3. Or manually break it into cells

## Still Having Issues?

If none of the above work:

1. **Check Python version:**
   ```python
   import sys
   print(f"Python {sys.version}")
   ```

2. **Try a fresh Colab runtime:**
   - Runtime -> Factory reset runtime
   - This will delete all installed packages and start fresh

3. **Use a local environment:**
   - Download the file and run it locally with Python 3.10 or 3.11
   - Colab's Python 3.12 may have compatibility issues

## Quick Test

To verify PyTorch is working:

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"autograd available: {hasattr(torch, 'autograd')}")
```

If the last line prints `True`, PyTorch is working correctly.
