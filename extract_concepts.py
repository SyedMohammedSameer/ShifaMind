#!/usr/bin/env python3
"""
Extract the 150 concepts from 016.py training setup
Run this ONCE in Colab to create concepts_for_demo.pkl

Then demo1.py can load this file instantly without UMLS loading.
"""

import pickle
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

# Paths (Colab)
BASE_PATH = Path('/content/drive/MyDrive/ShifaMind/01_Raw_Datasets/Extracted')
UMLS_PATH = BASE_PATH / 'umls-2025AA-metathesaurus-full/2025AA/META'

print("="*70)
print("EXTRACT 150 CONCEPTS FOR DEMO")
print("="*70)

# ============================================================================
# Load minimal UMLS data
# ============================================================================
class FastUMLSLoader:
    def __init__(self, umls_path):
        self.umls_path = umls_path
        self.concepts = {}
        self.icd10_to_cui = defaultdict(list)

    def load_concepts(self, max_concepts=30000):
        print(f"\n📚 Loading UMLS concepts (max: {max_concepts})...")

        target_types = {'T047', 'T046', 'T184', 'T033', 'T048', 'T037', 'T191', 'T020'}
        cui_to_types = self._load_semantic_types()

        mrconso_path = self.umls_path / 'MRCONSO.RRF'
        concepts_loaded = 0

        print("  Loading MRCONSO...")
        with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="  Parsing", total=max_concepts):
                if concepts_loaded >= max_concepts:
                    break

                fields = line.strip().split('|')
                if len(fields) < 15:
                    continue

                cui = fields[0]
                lang = fields[1]
                sab = fields[11]
                code = fields[13]
                term = fields[14]

                if lang != 'ENG':
                    continue
                if sab not in ['SNOMEDCT_US', 'ICD10CM', 'MSH', 'NCI']:
                    continue

                if cui not in cui_to_types:
                    continue
                types = cui_to_types[cui]
                if not any(t in target_types for t in types):
                    continue

                if cui not in self.concepts:
                    self.concepts[cui] = {
                        'cui': cui,
                        'name': term,
                        'terms': [term],
                        'sources': {sab: [code]},
                        'semantic_types': types
                    }
                    concepts_loaded += 1
                else:
                    if term not in self.concepts[cui]['terms']:
                        self.concepts[cui]['terms'].append(term)

                if sab == 'ICD10CM' and code:
                    self.icd10_to_cui[code].append(cui)

        print(f"  ✅ Loaded {len(self.concepts)} concepts")
        return self.concepts

    def _load_semantic_types(self):
        mrsty_path = self.umls_path / 'MRSTY.RRF'
        cui_to_types = defaultdict(list)

        with open(mrsty_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) >= 2:
                    cui_to_types[fields[0]].append(fields[1])

        return cui_to_types

    def load_definitions(self, concepts):
        print("\n📖 Loading definitions...")
        mrdef_path = self.umls_path / 'MRDEF.RRF'
        definitions_added = 0

        with open(mrdef_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="  Parsing"):
                fields = line.strip().split('|')
                if len(fields) >= 6:
                    cui = fields[0]
                    definition = fields[5]

                    if cui in concepts and definition:
                        if 'definition' not in concepts[cui]:
                            concepts[cui]['definition'] = definition
                            definitions_added += 1

        print(f"  ✅ Added {definitions_added} definitions")
        return concepts

# ============================================================================
# Build concept set (same as 016.py)
# ============================================================================
class SemanticTypeValidator:
    RELEVANT_TYPES = {'T047', 'T046', 'T184', 'T033', 'T048', 'T037', 'T191', 'T020'}

    DIAGNOSIS_SEMANTIC_GROUPS = {
        'J': {'T047', 'T046', 'T184', 'T033'},
        'I': {'T047', 'T046', 'T184', 'T033'},
        'A': {'T047', 'T046', 'T184', 'T033'},
        'K': {'T047', 'T046', 'T184', 'T033'},
    }

    def __init__(self, umls_concepts):
        self.umls_concepts = umls_concepts

    def validate_concept(self, cui, diagnosis_code=None):
        if cui not in self.umls_concepts:
            return False

        concept = self.umls_concepts[cui]
        semantic_types = set(concept.get('semantic_types', []))

        if not semantic_types.intersection(self.RELEVANT_TYPES):
            return False

        if diagnosis_code:
            prefix = diagnosis_code[0]
            expected_types = self.DIAGNOSIS_SEMANTIC_GROUPS.get(prefix, self.RELEVANT_TYPES)
            if not semantic_types.intersection(expected_types):
                return False

        return True

def build_150_concepts(umls_concepts, icd_to_cui):
    """Build the same 150 concepts as 016.py"""

    print("\n🔬 Building 150 concept set...")

    target_codes = ['J189', 'I5023', 'A419', 'K8000']
    semantic_validator = SemanticTypeValidator(umls_concepts)
    relevant_cuis = set()

    # Strategy 1: Direct ICD mappings
    def get_icd_variants(code):
        variants = {code, code.replace('.', '')}
        no_dots = code.replace('.', '')
        if len(no_dots) >= 4:
            variants.add(no_dots[:3] + '.' + no_dots[3:])
        variants.add(no_dots[:3])
        return list(variants)

    for icd in target_codes:
        variants = get_icd_variants(icd)
        for variant in variants:
            if variant in icd_to_cui:
                cuis = icd_to_cui[variant]
                validated = [
                    cui for cui in cuis
                    if semantic_validator.validate_concept(cui, icd)
                ]
                relevant_cuis.update(validated[:30])

    print(f"  Direct mappings: {len(relevant_cuis)} concepts")

    # Strategy 2: Keyword expansion
    diagnosis_keywords = {
        'J189': ['pneumonia', 'lung infection', 'respiratory infection',
                 'infiltrate', 'bacterial pneumonia', 'aspiration'],
        'I5023': ['heart failure', 'cardiac failure', 'cardiomyopathy',
                  'pulmonary edema', 'ventricular dysfunction'],
        'A419': ['sepsis', 'septicemia', 'bacteremia', 'infection',
                 'septic shock', 'organ dysfunction'],
        'K8000': ['cholecystitis', 'gallbladder', 'biliary disease',
                  'gallstone', 'cholelithiasis']
    }

    for icd in target_codes:
        keywords = diagnosis_keywords.get(icd, [])

        for cui, info in umls_concepts.items():
            if cui in relevant_cuis:
                continue

            terms_text = ' '.join([info['name']] + info.get('terms', [])).lower()

            if any(kw in terms_text for kw in keywords):
                if semantic_validator.validate_concept(cui, icd):
                    relevant_cuis.add(cui)

            if len(relevant_cuis) >= 150:
                break

        if len(relevant_cuis) >= 150:
            break

    print(f"  After expansion: {len(relevant_cuis)} concepts")

    # Build final concept dictionary
    final_concepts = {}
    for cui in list(relevant_cuis)[:150]:
        if cui in umls_concepts:
            concept = umls_concepts[cui]
            final_concepts[cui] = {
                'cui': cui,
                'name': concept['name'],
                'definition': concept.get('definition', ''),
                'terms': concept.get('terms', []),
                'semantic_types': concept.get('semantic_types', [])
            }

    print(f"  ✅ Final: {len(final_concepts)} concepts")

    return final_concepts

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # Load UMLS
    print("\n📂 Loading UMLS...")
    umls_loader = FastUMLSLoader(UMLS_PATH)
    umls_concepts = umls_loader.load_concepts(max_concepts=30000)
    umls_concepts = umls_loader.load_definitions(umls_concepts)

    # Build 150 concepts
    final_concepts = build_150_concepts(umls_concepts, umls_loader.icd10_to_cui)

    # Save for demo
    output_file = 'concepts_for_demo.pkl'
    print(f"\n💾 Saving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(final_concepts, f)

    print(f"✅ Saved {len(final_concepts)} concepts")

    # Also save as JSON for inspection
    import json
    json_file = 'concepts_for_demo.json'
    print(f"💾 Saving to {json_file}...")
    with open(json_file, 'w') as f:
        json.dump(final_concepts, f, indent=2)

    print("\n" + "="*70)
    print("✅ CONCEPT EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  - {output_file} (for demo to load)")
    print(f"  - {json_file} (for inspection)")
    print(f"\nNext step:")
    print(f"  Upload {output_file} to Colab with demo1.py")
    print("="*70)
