"""
ShifaMind - Colab Setup Script

Run this first in Google Colab to set up the folder structure.
This will create all necessary directories and show you what files to upload.

Usage:
    !python setup_colab.py
"""

from pathlib import Path
import os

def setup_colab_environment():
    """Set up folder structure for ShifaMind in Colab"""

    print("=" * 80)
    print("SHIFAMIND - GOOGLE COLAB SETUP")
    print("=" * 80)
    print()

    # Create base directory
    base = Path('/content/ShifaMind_Data')

    # Create all necessary folders
    folders = {
        'UMLS': base / 'UMLS',
        'ICD10': base / 'ICD10',
        'MIMIC': base / 'MIMIC',
        'Models': base / 'Models',
        'Results': base / 'Results',
    }

    print("📁 Creating folder structure...")
    for name, path in folders.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {path}")

    print()
    print("=" * 80)
    print("FOLDER STRUCTURE READY!")
    print("=" * 80)
    print()
    print("Now upload your data files to these locations:")
    print()
    print("📋 STEP 1: Upload UMLS files to /content/ShifaMind_Data/UMLS/")
    print("  Required files:")
    print("    • MRCONSO.RRF")
    print("    • MRSTY.RRF")
    print("    • MRDEF.RRF (optional)")
    print()
    print("📋 STEP 2: Upload ICD-10 file to /content/ShifaMind_Data/ICD10/")
    print("  Required file:")
    print("    • icd10cm-codes-2024.txt")
    print()
    print("📋 STEP 3: Upload MIMIC-IV files to /content/ShifaMind_Data/MIMIC/")
    print("  Required files:")
    print("    • noteevents.csv.gz")
    print("    • diagnoses_icd.csv.gz")
    print("    • patients.csv.gz")
    print("    • admissions.csv.gz")
    print()
    print("=" * 80)
    print("HOW TO UPLOAD:")
    print("=" * 80)
    print()
    print("Option A - Using Colab File Browser:")
    print("  1. Click the folder icon 📁 on the left sidebar")
    print("  2. Navigate to /content/ShifaMind_Data/UMLS/")
    print("  3. Click the upload button and select your UMLS files")
    print("  4. Repeat for ICD10 and MIMIC folders")
    print()
    print("Option B - Using Python upload:")
    print("  Run this code:")
    print()
    print("  from google.colab import files")
    print("  import shutil")
    print()
    print("  # Upload UMLS files")
    print("  print('Upload MRCONSO.RRF:')")
    print("  uploaded = files.upload()")
    print("  shutil.move(list(uploaded.keys())[0], '/content/ShifaMind_Data/UMLS/')")
    print()
    print("=" * 80)
    print()
    print("After uploading, run:")
    print("  !python final_knowledge_base_generator.py")
    print()
    print("=" * 80)

    # Check if files already exist
    print()
    print("🔍 Checking for existing data files...")
    print()

    umls_files = {
        'MRCONSO.RRF': folders['UMLS'] / 'MRCONSO.RRF',
        'MRSTY.RRF': folders['UMLS'] / 'MRSTY.RRF',
    }

    icd10_files = {
        'icd10cm-codes-2024.txt': folders['ICD10'] / 'icd10cm-codes-2024.txt',
    }

    mimic_files = {
        'noteevents.csv.gz': folders['MIMIC'] / 'noteevents.csv.gz',
        'diagnoses_icd.csv.gz': folders['MIMIC'] / 'diagnoses_icd.csv.gz',
    }

    all_ready = True

    for name, path in umls_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {name} - NOT FOUND")
            all_ready = False

    for name, path in icd10_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {name} - NOT FOUND")
            all_ready = False

    for name, path in mimic_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠️  {name} - NOT FOUND (needed for training)")

    print()

    if all_ready:
        print("=" * 80)
        print("🎉 ALL REQUIRED FILES FOUND! You can now run:")
        print("  !python final_knowledge_base_generator.py")
        print("=" * 80)
    else:
        print("=" * 80)
        print("⚠️  Please upload missing files before continuing")
        print("=" * 80)

    print()

if __name__ == '__main__':
    setup_colab_environment()
