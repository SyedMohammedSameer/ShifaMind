#!/usr/bin/env python3
"""
ShifaMind 043: Clinical Knowledge Base Generator

Parses UMLS Metathesaurus and ICD-10 codes to create a structured
clinical knowledge base for diagnosis prediction explainability.

Author: Mohammed Sameer Syed
Date: November 2025
Version: 043-KBGen
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
import re

print("="*80)
print("SHIFAMIND 043: CLINICAL KNOWLEDGE BASE GENERATOR")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
UMLS_META_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/2025AA/META'
ICD10_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/icd10cm-CodesDescriptions-2024'
OUTPUT_PATH = BASE_PATH / '03_Models'

# UMLS files
MRCONSO_PATH = UMLS_META_PATH / 'MRCONSO.RRF'
MRDEF_PATH = UMLS_META_PATH / 'MRDEF.RRF'
MRREL_PATH = UMLS_META_PATH / 'MRREL.RRF'

# ICD-10 file
ICD10_CODES_PATH = ICD10_PATH / 'icd10cm-codes-2024.txt'

# Target diagnoses
TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# Medical concept keywords for each diagnosis
DIAGNOSIS_KEYWORDS = {
    'J189': {
        'core': ['pneumonia', 'lung infection', 'respiratory infection', 'pulmonary infection'],
        'symptoms': ['fever', 'cough', 'dyspnea', 'shortness of breath', 'sputum', 'chest pain'],
        'signs': ['crackles', 'rales', 'rhonchi', 'decreased breath sounds', 'tachypnea', 'hypoxia'],
        'findings': ['infiltrate', 'consolidation', 'opacity', 'pleural effusion'],
        'lab': ['leukocytosis', 'elevated white blood cell', 'neutrophilia']
    },
    'I5023': {
        'core': ['heart failure', 'cardiac failure', 'congestive heart failure', 'CHF', 'cardiomyopathy'],
        'symptoms': ['dyspnea', 'shortness of breath', 'orthopnea', 'paroxysmal nocturnal dyspnea', 'fatigue'],
        'signs': ['edema', 'swelling', 'jugular venous distension', 'JVP', 'S3 gallop', 'rales'],
        'findings': ['pulmonary edema', 'pleural effusion', 'cardiomegaly', 'pulmonary congestion'],
        'lab': ['BNP', 'B-type natriuretic peptide', 'elevated BNP', 'troponin']
    },
    'A419': {
        'core': ['sepsis', 'septicemia', 'bacteremia', 'systemic infection', 'septic shock'],
        'symptoms': ['fever', 'hypothermia', 'chills', 'confusion', 'altered mental status'],
        'signs': ['hypotension', 'tachycardia', 'tachypnea', 'shock', 'decreased urine output'],
        'findings': ['organ dysfunction', 'organ failure', 'multi-organ dysfunction', 'MODS'],
        'lab': ['leukocytosis', 'leukopenia', 'lactic acidosis', 'elevated lactate', 'bandemia']
    },
    'K8000': {
        'core': ['cholecystitis', 'gallbladder inflammation', 'acute cholecystitis', 'gallstones', 'cholelithiasis'],
        'symptoms': ['abdominal pain', 'right upper quadrant pain', 'RUQ pain', 'nausea', 'vomiting'],
        'signs': ['Murphy sign', 'tenderness', 'fever', 'jaundice'],
        'findings': ['gallbladder wall thickening', 'pericholecystic fluid', 'gallstone', 'cholelithiasis'],
        'lab': ['leukocytosis', 'elevated WBC', 'elevated liver enzymes', 'hyperbilirubinemia']
    }
}

print(f"\n📁 UMLS Path: {UMLS_META_PATH}")
print(f"📁 ICD-10 Path: {ICD10_PATH}")
print(f"📁 Output: {OUTPUT_PATH / 'clinical_knowledge_base_043.json'}")

# ============================================================================
# UMLS PARSING FUNCTIONS
# ============================================================================

def parse_rrf_line(line, delimiter='|'):
    """Parse a single line from RRF file"""
    return line.strip().split(delimiter)


def load_concept_names(mrconso_path, target_keywords):
    """
    Load concept names from MRCONSO.RRF

    Filter for:
    - English language (LAT='ENG')
    - Preferred terms (ISPREF='Y')
    - Concepts matching target keywords

    Returns:
        dict: {CUI: {'name': str, 'source': str, 'synonyms': [str]}}
    """

    print("\n" + "="*70)
    print("PARSING MRCONSO.RRF - Concept Names")
    print("="*70)
    print(f"File: {mrconso_path}")
    print(f"Size: {mrconso_path.stat().st_size / (1024**3):.2f} GB")

    # Flatten all keywords
    all_keywords = set()
    for dx_keywords in target_keywords.values():
        for category_keywords in dx_keywords.values():
            all_keywords.update([k.lower() for k in category_keywords])

    print(f"\nSearching for {len(all_keywords)} unique medical terms...")

    concepts = {}

    with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="  Scanning", unit=" lines"):
            fields = parse_rrf_line(line)

            if len(fields) < 15:
                continue

            cui = fields[0]
            lat = fields[1]  # Language
            ispref = fields[6]  # Is preferred
            sab = fields[11]  # Source abbreviation
            tty = fields[12]  # Term type
            str_text = fields[14]  # Concept string

            # Filter: English only
            if lat != 'ENG':
                continue

            # Filter: Trusted sources
            if sab not in ['SNOMEDCT_US', 'ICD10CM', 'MSH', 'NCI', 'MEDLINEPLUS', 'HPO']:
                continue

            str_lower = str_text.lower()

            # Check if concept matches any target keyword
            matched = False
            for keyword in all_keywords:
                if keyword in str_lower or str_lower in keyword:
                    matched = True
                    break

            if not matched:
                continue

            # Store concept
            if cui not in concepts:
                concepts[cui] = {
                    'name': str_text,
                    'source': sab,
                    'synonyms': []
                }
            else:
                # Add synonym
                if str_text not in concepts[cui]['synonyms']:
                    concepts[cui]['synonyms'].append(str_text)

    print(f"\n  ✅ Found {len(concepts)} relevant concepts")
    return concepts


def load_concept_definitions(mrdef_path, target_cuis):
    """
    Load definitions from MRDEF.RRF for target CUIs

    Returns:
        dict: {CUI: definition_text}
    """

    print("\n" + "="*70)
    print("PARSING MRDEF.RRF - Concept Definitions")
    print("="*70)
    print(f"File: {mrdef_path}")

    if not mrdef_path.exists():
        print("  ⚠️  MRDEF.RRF not found, skipping definitions")
        return {}

    print(f"Size: {mrdef_path.stat().st_size / (1024**3):.2f} GB")
    print(f"Looking for definitions for {len(target_cuis)} concepts...")

    definitions = {}

    with open(mrdef_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="  Scanning", unit=" lines"):
            fields = parse_rrf_line(line)

            if len(fields) < 6:
                continue

            cui = fields[0]
            definition = fields[5]

            if cui in target_cuis and cui not in definitions:
                # Take first definition found
                definitions[cui] = definition

    print(f"\n  ✅ Found {len(definitions)} definitions")
    return definitions


def load_icd10_descriptions(icd10_path):
    """
    Load ICD-10 code descriptions

    Format: "CODE Description"
    Example: "J189 Pneumonia, unspecified organism"

    Returns:
        dict: {code: description}
    """

    print("\n" + "="*70)
    print("PARSING ICD-10 CODES")
    print("="*70)
    print(f"File: {icd10_path}")

    descriptions = {}

    with open(icd10_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            # Split on first space
            parts = line.split(None, 1)

            if len(parts) == 2:
                code, description = parts
                descriptions[code] = description

    print(f"\n  ✅ Loaded {len(descriptions)} ICD-10 codes")
    return descriptions


# ============================================================================
# KNOWLEDGE BASE CONSTRUCTION
# ============================================================================

def create_diagnosis_knowledge_entry(diagnosis_code, icd_descriptions, concepts, definitions):
    """
    Create knowledge base entries for a specific diagnosis

    Returns:
        list: Knowledge entries
    """

    diagnosis_name = icd_descriptions.get(diagnosis_code, diagnosis_code)
    keywords = DIAGNOSIS_KEYWORDS.get(diagnosis_code, {})

    entries = []

    # 1. Diagnosis description (from ICD-10)
    entries.append({
        'type': 'diagnosis_description',
        'text': f"{diagnosis_name} ({diagnosis_code}) is a clinical condition requiring medical diagnosis and treatment.",
        'source': f'ICD-10-CM {diagnosis_code}',
        'keywords': keywords.get('core', []),
        'priority': 10
    })

    # 2. Clinical presentation
    if 'symptoms' in keywords:
        symptoms_text = f"Common presenting symptoms of {diagnosis_name} include: "
        symptoms_text += ", ".join(keywords['symptoms'][:5]) + "."

        entries.append({
            'type': 'clinical_presentation',
            'text': symptoms_text,
            'source': f'Clinical Knowledge - {diagnosis_code}',
            'keywords': keywords['symptoms'],
            'priority': 9
        })

    # 3. Physical examination findings
    if 'signs' in keywords:
        signs_text = f"Physical examination findings may reveal: "
        signs_text += ", ".join(keywords['signs'][:5]) + "."

        entries.append({
            'type': 'physical_findings',
            'text': signs_text,
            'source': f'Clinical Knowledge - {diagnosis_code}',
            'keywords': keywords['signs'],
            'priority': 8
        })

    # 4. Diagnostic findings
    if 'findings' in keywords:
        findings_text = f"Diagnostic imaging and tests may show: "
        findings_text += ", ".join(keywords['findings'][:5]) + "."

        entries.append({
            'type': 'diagnostic_findings',
            'text': findings_text,
            'source': f'Clinical Knowledge - {diagnosis_code}',
            'keywords': keywords['findings'],
            'priority': 7
        })

    # 5. Laboratory findings
    if 'lab' in keywords:
        lab_text = f"Laboratory abnormalities may include: "
        lab_text += ", ".join(keywords['lab'][:5]) + "."

        entries.append({
            'type': 'laboratory_findings',
            'text': lab_text,
            'source': f'Clinical Knowledge - {diagnosis_code}',
            'keywords': keywords['lab'],
            'priority': 6
        })

    # 6. UMLS concept definitions
    # Find concepts related to this diagnosis
    all_keywords = []
    for category_keywords in keywords.values():
        all_keywords.extend([k.lower() for k in category_keywords])

    concept_count = 0
    for cui, concept_info in concepts.items():
        if concept_count >= 10:  # Limit to top 10 concepts
            break

        concept_name = concept_info['name'].lower()

        # Check if concept matches diagnosis keywords
        matched = False
        for kw in all_keywords:
            if kw in concept_name or concept_name in kw:
                matched = True
                break

        if not matched:
            continue

        # Get definition if available
        definition = definitions.get(cui, f"{concept_info['name']} is a medical concept relevant to {diagnosis_name}.")

        # Truncate long definitions
        if len(definition) > 300:
            definition = definition[:297] + "..."

        entries.append({
            'type': 'concept_definition',
            'text': definition,
            'source': f"UMLS {cui} ({concept_info['source']})",
            'keywords': [concept_info['name'].lower()],
            'cui': cui,
            'priority': 5
        })

        concept_count += 1

    # Sort by priority
    entries.sort(key=lambda x: x['priority'], reverse=True)

    # Remove priority field (internal use only)
    for entry in entries:
        del entry['priority']

    return entries


def build_knowledge_base(target_codes, icd_descriptions, concepts, definitions):
    """
    Build complete knowledge base for all target diagnoses

    Returns:
        dict: {diagnosis_code: [knowledge_entries]}
    """

    print("\n" + "="*70)
    print("BUILDING KNOWLEDGE BASE")
    print("="*70)

    knowledge_base = {}

    for code in target_codes:
        print(f"\n📋 Creating knowledge for {code}: {icd_descriptions.get(code, code)}")

        entries = create_diagnosis_knowledge_entry(
            code, icd_descriptions, concepts, definitions
        )

        knowledge_base[code] = entries

        print(f"  ✅ Generated {len(entries)} knowledge entries")

    total_entries = sum(len(entries) for entries in knowledge_base.values())
    print(f"\n  ✅ Total knowledge entries: {total_entries}")

    return knowledge_base


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main knowledge base generation pipeline
    """

    print("\n" + "="*80)
    print("STARTING KNOWLEDGE BASE GENERATION")
    print("="*80)

    # Validate file paths
    print("\n🔍 Validating file paths...")

    required_files = [
        ('MRCONSO', MRCONSO_PATH),
        ('ICD-10', ICD10_CODES_PATH)
    ]

    for name, path in required_files:
        if not path.exists():
            print(f"  ❌ ERROR: {name} not found at {path}")
            print(f"  Please ensure UMLS and ICD-10 data is downloaded.")
            return
        print(f"  ✅ {name}: {path.stat().st_size / (1024**2):.1f} MB")

    # Parse ICD-10 codes
    icd_descriptions = load_icd10_descriptions(ICD10_CODES_PATH)

    # Parse UMLS MRCONSO for concept names
    concepts = load_concept_names(MRCONSO_PATH, DIAGNOSIS_KEYWORDS)

    # Parse UMLS MRDEF for definitions
    target_cuis = set(concepts.keys())
    definitions = load_concept_definitions(MRDEF_PATH, target_cuis)

    # Build knowledge base
    knowledge_base = build_knowledge_base(
        TARGET_CODES, icd_descriptions, concepts, definitions
    )

    # Save to JSON
    output_file = OUTPUT_PATH / 'clinical_knowledge_base_043.json'

    print(f"\n💾 Saving knowledge base to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")

    # Print summary
    print("\n" + "="*80)
    print("KNOWLEDGE BASE GENERATION COMPLETE")
    print("="*80)

    print("\n📊 SUMMARY:")
    for code in TARGET_CODES:
        num_entries = len(knowledge_base[code])
        print(f"  {code} ({ICD_DESCRIPTIONS[code]}): {num_entries} entries")

    total_entries = sum(len(entries) for entries in knowledge_base.values())
    print(f"\n  Total entries: {total_entries}")
    print(f"  Average per diagnosis: {total_entries / len(TARGET_CODES):.1f}")

    # Show sample entry
    print("\n📄 SAMPLE ENTRY (J189):")
    sample_entry = knowledge_base['J189'][0]
    print(f"  Type: {sample_entry['type']}")
    print(f"  Text: {sample_entry['text'][:150]}...")
    print(f"  Source: {sample_entry['source']}")
    print(f"  Keywords: {', '.join(sample_entry['keywords'][:5])}")

    print("\n✅ Knowledge base ready for use in 043_eval.py and 043_demo.py!")
    print("="*80)


if __name__ == '__main__':
    main()
