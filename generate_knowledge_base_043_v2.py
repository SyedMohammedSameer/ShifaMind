#!/usr/bin/env python3
"""
ShifaMind 043: Clinical Knowledge Base Generator (V2 - Filtered)

UPDATED: Filters out veterinary, animal, and non-human medical concepts

Author: Mohammed Sameer Syed
Date: November 2025
Version: 043-KBGen-V2-Filtered
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
import re

print("="*80)
print("SHIFAMIND 043: CLINICAL KNOWLEDGE BASE GENERATOR V2 (FILTERED)")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
UMLS_META_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/2025AA/META'
ICD10_PATH = BASE_PATH / '01_Raw_Datasets/Extracted/icd10cm-CodesDescriptions-2024'
OUTPUT_PATH = BASE_PATH / '03_Models'

MRCONSO_PATH = UMLS_META_PATH / 'MRCONSO.RRF'
MRDEF_PATH = UMLS_META_PATH / 'MRDEF.RRF'
ICD10_CODES_PATH = ICD10_PATH / 'icd10cm-codes-2024.txt'

TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']
ICD_DESCRIPTIONS = {
    'J189': 'Pneumonia, unspecified organism',
    'I5023': 'Acute on chronic systolic heart failure',
    'A419': 'Sepsis, unspecified organism',
    'K8000': 'Calculus of gallbladder with acute cholecystitis'
}

# ============================================================================
# FILTERING CONFIGURATION (NEW!)
# ============================================================================

# Exclusion keywords - if concept/definition contains these, skip it
ANIMAL_KEYWORDS = [
    'cattle', 'bovine', 'cow', 'cows', 'bull', 'bulls',
    'pig', 'pigs', 'swine', 'porcine', 'hog', 'hogs',
    'horse', 'horses', 'equine', 'foal',
    'sheep', 'ovine', 'lamb',
    'goat', 'goats', 'caprine',
    'dog', 'dogs', 'canine', 'puppy',
    'cat', 'cats', 'feline', 'kitten',
    'chicken', 'poultry', 'avian', 'bird',
    'fish', 'fishes', 'aquatic',
    'rodent', 'mouse', 'mice', 'rat', 'rats',
    'veterinary', 'veterinarian', 'animal', 'animals',
    'livestock', 'farm animal', 'domestic animal',
    'wildlife', 'zoo', 'zoological'
]

# Additional exclusion patterns
EXCLUSION_PATTERNS = [
    r'\bof cattle\b',
    r'\bof pigs\b', 
    r'\bof swine\b',
    r'\bin cattle\b',
    r'\bin pigs\b',
    r'\bin swine\b',
    r'\bANIMAL\b',
    r'\bVETERINARY\b'
]

def contains_animal_content(text):
    """Check if text contains animal/veterinary keywords"""
    text_lower = text.lower()
    
    # Check keywords
    for keyword in ANIMAL_KEYWORDS:
        if keyword in text_lower:
            return True
    
    # Check patterns
    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False

# ============================================================================
# MEDICAL CONCEPT KEYWORDS (HUMAN DISEASES ONLY)
# ============================================================================

DIAGNOSIS_KEYWORDS = {
    'J189': {
        'core': ['pneumonia', 'lung infection', 'respiratory infection', 'pulmonary infection', 'chest infection'],
        'symptoms': ['fever', 'cough', 'dyspnea', 'shortness of breath', 'sputum', 'chest pain', 'hemoptysis'],
        'signs': ['crackles', 'rales', 'rhonchi', 'decreased breath sounds', 'tachypnea', 'hypoxia', 'respiratory distress'],
        'findings': ['infiltrate', 'consolidation', 'opacity', 'pleural effusion', 'alveolar', 'interstitial'],
        'lab': ['leukocytosis', 'elevated white blood cell', 'neutrophilia', 'procalcitonin']
    },
    'I5023': {
        'core': ['heart failure', 'cardiac failure', 'congestive heart failure', 'CHF', 'cardiomyopathy', 'ventricular dysfunction'],
        'symptoms': ['dyspnea', 'shortness of breath', 'orthopnea', 'paroxysmal nocturnal dyspnea', 'fatigue', 'weakness'],
        'signs': ['edema', 'swelling', 'jugular venous distension', 'JVP', 'S3 gallop', 'rales', 'hepatomegaly'],
        'findings': ['pulmonary edema', 'pleural effusion', 'cardiomegaly', 'pulmonary congestion', 'reduced ejection fraction'],
        'lab': ['BNP', 'B-type natriuretic peptide', 'elevated BNP', 'NT-proBNP', 'troponin']
    },
    'A419': {
        'core': ['sepsis', 'septicemia', 'bacteremia', 'systemic infection', 'septic shock', 'SIRS'],
        'symptoms': ['fever', 'hypothermia', 'chills', 'rigors', 'confusion', 'altered mental status', 'lethargy'],
        'signs': ['hypotension', 'tachycardia', 'tachypnea', 'shock', 'decreased urine output', 'mottled skin'],
        'findings': ['organ dysfunction', 'organ failure', 'multi-organ dysfunction', 'MODS', 'hypoperfusion'],
        'lab': ['leukocytosis', 'leukopenia', 'lactic acidosis', 'elevated lactate', 'bandemia', 'procalcitonin']
    },
    'K8000': {
        'core': ['cholecystitis', 'gallbladder inflammation', 'acute cholecystitis', 'gallstones', 'cholelithiasis', 'biliary'],
        'symptoms': ['abdominal pain', 'right upper quadrant pain', 'RUQ pain', 'nausea', 'vomiting', 'anorexia'],
        'signs': ['Murphy sign', 'tenderness', 'fever', 'jaundice', 'icterus', 'guarding'],
        'findings': ['gallbladder wall thickening', 'pericholecystic fluid', 'gallstone', 'cholelithiasis', 'distended gallbladder'],
        'lab': ['leukocytosis', 'elevated WBC', 'elevated liver enzymes', 'hyperbilirubinemia', 'alkaline phosphatase']
    }
}

print(f"\n📁 UMLS Path: {UMLS_META_PATH}")
print(f"📁 ICD-10 Path: {ICD10_PATH}")
print(f"📁 Output: {OUTPUT_PATH / 'clinical_knowledge_base_043.json'}")
print(f"\n🚫 Filtering: {len(ANIMAL_KEYWORDS)} animal keywords, {len(EXCLUSION_PATTERNS)} patterns")

# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_rrf_line(line, delimiter='|'):
    """Parse a single line from RRF file"""
    return line.strip().split(delimiter)


def load_concept_names(mrconso_path, target_keywords):
    """
    Load concept names from MRCONSO.RRF with animal filtering
    """

    print("\n" + "="*70)
    print("PARSING MRCONSO.RRF - Concept Names (WITH FILTERING)")
    print("="*70)
    print(f"File: {mrconso_path}")
    print(f"Size: {mrconso_path.stat().st_size / (1024**3):.2f} GB")

    all_keywords = set()
    for dx_keywords in target_keywords.values():
        for category_keywords in dx_keywords.values():
            all_keywords.update([k.lower() for k in category_keywords])

    print(f"\nSearching for {len(all_keywords)} unique medical terms...")
    print(f"Filtering out animal/veterinary concepts...")

    concepts = {}
    filtered_count = 0

    with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="  Scanning", unit=" lines"):
            fields = parse_rrf_line(line)

            if len(fields) < 15:
                continue

            cui = fields[0]
            lat = fields[1]
            ispref = fields[6]
            sab = fields[11]
            str_text = fields[14]

            if lat != 'ENG':
                continue

            if sab not in ['SNOMEDCT_US', 'ICD10CM', 'MSH', 'NCI', 'MEDLINEPLUS', 'HPO']:
                continue

            # FILTER: Check for animal content
            if contains_animal_content(str_text):
                filtered_count += 1
                continue

            str_lower = str_text.lower()

            matched = False
            for keyword in all_keywords:
                if keyword in str_lower or str_lower in keyword:
                    matched = True
                    break

            if not matched:
                continue

            if cui not in concepts:
                concepts[cui] = {
                    'name': str_text,
                    'source': sab,
                    'synonyms': []
                }
            else:
                if str_text not in concepts[cui]['synonyms']:
                    concepts[cui]['synonyms'].append(str_text)

    print(f"\n  ✅ Found {len(concepts)} relevant concepts")
    print(f"  🚫 Filtered out {filtered_count} animal/veterinary concepts")
    return concepts


def load_concept_definitions(mrdef_path, target_cuis):
    """
    Load definitions from MRDEF.RRF with animal filtering
    """

    print("\n" + "="*70)
    print("PARSING MRDEF.RRF - Concept Definitions (WITH FILTERING)")
    print("="*70)
    print(f"File: {mrdef_path}")

    if not mrdef_path.exists():
        print("  ⚠️  MRDEF.RRF not found, skipping definitions")
        return {}

    print(f"Size: {mrdef_path.stat().st_size / (1024**3):.2f} GB")
    print(f"Looking for definitions for {len(target_cuis)} concepts...")

    definitions = {}
    filtered_count = 0

    with open(mrdef_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="  Scanning", unit=" lines"):
            fields = parse_rrf_line(line)

            if len(fields) < 6:
                continue

            cui = fields[0]
            definition = fields[5]

            if cui not in target_cuis:
                continue

            # FILTER: Check for animal content in definition
            if contains_animal_content(definition):
                filtered_count += 1
                continue

            if cui not in definitions:
                definitions[cui] = definition

    print(f"\n  ✅ Found {len(definitions)} definitions")
    print(f"  🚫 Filtered out {filtered_count} animal/veterinary definitions")
    return definitions


def load_icd10_descriptions(icd10_path):
    """Load ICD-10 code descriptions"""

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
    """Create knowledge base entries for a specific diagnosis"""

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

    # 6. UMLS concept definitions (FILTERED for human medicine only)
    all_keywords = []
    for category_keywords in keywords.values():
        all_keywords.extend([k.lower() for k in category_keywords])

    concept_count = 0
    for cui, concept_info in concepts.items():
        if concept_count >= 10:
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

        # Get definition if available (already filtered for animals)
        definition = definitions.get(cui)
        
        if not definition:
            # Create generic definition
            definition = f"{concept_info['name']} is a medical concept relevant to {diagnosis_name}."

        # Double-check for animal content (safety check)
        if contains_animal_content(definition):
            continue

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

    # Remove priority field
    for entry in entries:
        del entry['priority']

    return entries


def build_knowledge_base(target_codes, icd_descriptions, concepts, definitions):
    """Build complete knowledge base for all target diagnoses"""

    print("\n" + "="*70)
    print("BUILDING KNOWLEDGE BASE (FILTERED)")
    print("="*70)

    knowledge_base = {}

    for code in target_codes:
        print(f"\n📋 Creating knowledge for {code}: {icd_descriptions.get(code, code)}")

        entries = create_diagnosis_knowledge_entry(
            code, icd_descriptions, concepts, definitions
        )

        knowledge_base[code] = entries

        print(f"  ✅ Generated {len(entries)} knowledge entries (all human medicine)")

    total_entries = sum(len(entries) for entries in knowledge_base.values())
    print(f"\n  ✅ Total knowledge entries: {total_entries}")

    return knowledge_base


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main knowledge base generation pipeline with filtering"""

    print("\n" + "="*80)
    print("STARTING FILTERED KNOWLEDGE BASE GENERATION")
    print("="*80)

    print("\n🔍 Validating file paths...")

    required_files = [
        ('MRCONSO', MRCONSO_PATH),
        ('ICD-10', ICD10_CODES_PATH)
    ]

    for name, path in required_files:
        if not path.exists():
            print(f"  ❌ ERROR: {name} not found at {path}")
            return
        print(f"  ✅ {name}: {path.stat().st_size / (1024**2):.1f} MB")

    # Parse ICD-10 codes
    icd_descriptions = load_icd10_descriptions(ICD10_CODES_PATH)

    # Parse UMLS MRCONSO (WITH FILTERING)
    concepts = load_concept_names(MRCONSO_PATH, DIAGNOSIS_KEYWORDS)

    # Parse UMLS MRDEF (WITH FILTERING)
    target_cuis = set(concepts.keys())
    definitions = load_concept_definitions(MRDEF_PATH, target_cuis)

    # Build knowledge base
    knowledge_base = build_knowledge_base(
        TARGET_CODES, icd_descriptions, concepts, definitions
    )

    # Save to JSON
    output_file = OUTPUT_PATH / 'clinical_knowledge_base_043.json'

    print(f"\n💾 Saving filtered knowledge base to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")

    # Print summary
    print("\n" + "="*80)
    print("FILTERED KNOWLEDGE BASE GENERATION COMPLETE")
    print("="*80)

    print("\n📊 SUMMARY:")
    for code in TARGET_CODES:
        num_entries = len(knowledge_base[code])
        print(f"  {code} ({ICD_DESCRIPTIONS[code]}): {num_entries} entries")

    total_entries = sum(len(entries) for entries in knowledge_base.values())
    print(f"\n  Total entries: {total_entries}")
    print(f"  Average per diagnosis: {total_entries / len(TARGET_CODES):.1f}")
    print(f"\n  🚫 All animal/veterinary content filtered out")

    # Show sample entry
    print("\n📄 SAMPLE ENTRY (J189):")
    sample_entry = knowledge_base['J189'][0]
    print(f"  Type: {sample_entry['type']}")
    print(f"  Text: {sample_entry['text'][:150]}...")
    print(f"  Source: {sample_entry['source']}")
    print(f"  Keywords: {', '.join(sample_entry['keywords'][:5])}")

    print("\n✅ Filtered knowledge base ready for use!")
    print("="*80)


if __name__ == '__main__':
    main()
