#!/usr/bin/env python3
"""
Colab Runner for 041.py - Safe execution wrapper

This wrapper ensures proper module initialization in Google Colab and
provides better error handling.

Usage in Colab:
    # Cell 1: Fix any PyTorch issues first
    !python COLAB_FIX.py

    # Cell 2: Run the main script
    %run 041.py

Alternative usage:
    !python run_041_colab.py
"""

import sys
import os

def check_environment():
    """Check if the environment is properly set up"""
    print("="*80)
    print("ENVIRONMENT CHECK")
    print("="*80)

    # Check Python version
    print(f"\n🐍 Python: {sys.version}")

    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")

        if not hasattr(torch, 'autograd'):
            print("❌ PyTorch autograd not available - circular import detected!")
            print("\n💡 Fix: Run COLAB_FIX.py first:")
            print("   !python COLAB_FIX.py")
            return False
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        print("\n💡 Fix: Run COLAB_FIX.py first:")
        print("   !python COLAB_FIX.py")
        return False

    # Check other dependencies
    missing = []
    for pkg in ['transformers', 'sklearn', 'pandas', 'tqdm']:
        try:
            __import__(pkg)
            print(f"✅ {pkg}: installed")
        except ImportError:
            missing.append(pkg)
            print(f"❌ {pkg}: NOT installed")

    if missing:
        print(f"\n💡 Install missing packages:")
        print(f"   !pip install {' '.join(missing)}")
        return False

    return True

def main():
    print("="*80)
    print("SHIFAMIND 041 - COLAB RUNNER")
    print("="*80)
    print("\n📋 This script helps you run 041.py safely in Google Colab")
    print("\n")

    # Check environment
    if not check_environment():
        print("\n" + "="*80)
        print("❌ ENVIRONMENT NOT READY")
        print("="*80)
        print("\nPlease fix the issues above before running 041.py")
        return 1

    print("\n" + "="*80)
    print("✅ ENVIRONMENT READY")
    print("="*80)

    print("\n📋 Recommended way to run 041.py in Colab:")
    print("\n   In a Colab cell, use:")
    print("   %run 041.py")
    print("\n   Or:")
    print("   !python 041.py")

    print("\n💡 Note: The script will take several hours to complete.")
    print("   It will train models in multiple stages.")

    user_input = input("\n❓ Run 041.py now? (yes/no): ").strip().lower()

    if user_input in ['yes', 'y']:
        print("\n" + "="*80)
        print("RUNNING 041.PY")
        print("="*80 + "\n")

        try:
            # Change to script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)

            # Run the script
            exec(open('041.py').read(), {'__name__': '__main__'})

            print("\n" + "="*80)
            print("✅ SCRIPT COMPLETED")
            print("="*80)
            return 0

        except Exception as e:
            print(f"\n❌ Error running 041.py: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n✅ Ready when you are! Run the script manually when ready.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
