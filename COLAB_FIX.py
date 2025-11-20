#!/usr/bin/env python3
"""
Automatic Fix for PyTorch Circular Import Error in Google Colab

This script fixes the AttributeError: partially initialized module 'torch' error.

Usage:
    1. Upload this file to your Colab environment
    2. Run: !python COLAB_FIX.py
    3. Follow the prompts
    4. Restart runtime when asked
    5. Run your 041.py script
"""

import sys
import subprocess
import os

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run a shell command and print results"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout:
                print(f"   Output: {result.stdout[:200]}")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"📌 Python Version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor == 12:
        print("⚠️  Warning: Python 3.12 detected!")
        print("   PyTorch may have compatibility issues with Python 3.12")
        print("   Recommended: Use Python 3.10 or 3.11")
        return False
    elif version.major == 3 and version.minor >= 10:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version too old, need 3.10+")
        return False

def test_torch_import():
    """Test if torch can be imported"""
    print("\n🧪 Testing PyTorch import...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully!")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   autograd available: {hasattr(torch, 'autograd')}")
        return True
    except AttributeError as e:
        print(f"❌ AttributeError: {e}")
        print("   This is the circular import error we need to fix")
        return False
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def clear_caches():
    """Clear Python caches"""
    print("\n🧹 Clearing Python caches...")

    # Clear __pycache__ directories
    run_command(
        "find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true",
        "Clearing __pycache__ directories"
    )

    # Clear .pyc files
    run_command(
        "find . -type f -name '*.pyc' -delete 2>/dev/null || true",
        "Deleting .pyc files"
    )

    # Clear imports from sys.modules
    if 'torch' in sys.modules:
        print("🔧 Removing torch from sys.modules...")
        del sys.modules['torch']
        print("✅ Cleared torch from cache")

def reinstall_pytorch():
    """Reinstall PyTorch"""
    print("\n📦 Reinstalling PyTorch...")
    print("   This may take 1-2 minutes...\n")

    # Uninstall
    success = run_command(
        "pip uninstall -y torch torchvision torchaudio",
        "Uninstalling PyTorch"
    )

    if not success:
        print("⚠️  Uninstall had issues, continuing anyway...")

    # Install
    success = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "Installing PyTorch with CUDA 11.8"
    )

    return success

def main():
    print_header("COLAB PYTORCH CIRCULAR IMPORT FIX")

    print("This script will fix the error:")
    print("  'AttributeError: partially initialized module torch has no attribute autograd'\n")

    # Step 1: Check Python version
    print_header("STEP 1: Check Python Version")
    py_ok = check_python_version()

    # Step 2: Test current torch import
    print_header("STEP 2: Test Current PyTorch")
    torch_works = test_torch_import()

    if torch_works:
        print("\n✅ PyTorch is already working! No fix needed.")
        return 0

    # Step 3: Clear caches
    print_header("STEP 3: Clear Caches")
    clear_caches()

    # Step 4: Reinstall PyTorch
    print_header("STEP 4: Reinstall PyTorch")
    reinstall_success = reinstall_pytorch()

    if not reinstall_success:
        print("\n❌ Failed to reinstall PyTorch")
        print("\n💡 Manual fix steps:")
        print("   1. Runtime -> Restart runtime")
        print("   2. Run: !pip install --force-reinstall torch")
        print("   3. Try running your script again")
        return 1

    # Step 5: Final test
    print_header("STEP 5: Final Test")

    # Clear the module again before testing
    if 'torch' in sys.modules:
        del sys.modules['torch']

    torch_works = test_torch_import()

    if torch_works:
        print("\n" + "="*70)
        print("✅ SUCCESS! PyTorch is now working correctly!")
        print("="*70)
        print("\n📋 Next steps:")
        print("   1. **IMPORTANT: Restart the Colab runtime**")
        print("      Runtime -> Restart runtime")
        print("   2. Run your 041.py script:")
        print("      %run 041.py")
        print("\n")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️  PyTorch still has issues")
        print("="*70)
        print("\n💡 Additional steps to try:")
        print("   1. Runtime -> Factory reset runtime")
        print("   2. Re-run this fix script")
        print("   3. If still failing, the issue may be Python 3.12 incompatibility")
        print("\n")
        if not py_ok:
            print("⚠️  Your Python version (3.12) may be the root cause")
            print("   Consider using a local environment with Python 3.10 or 3.11")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Fix interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
