#!/usr/bin/env python3
"""
ShifaMind 043: Concept Filter Utility

Post-processes predicted concepts to remove veterinary/animal content
Applied at inference time (doesn't require retraining)

Author: Mohammed Sameer Syed
Date: November 2025
"""

import re

# ============================================================================
# FILTERING CONFIGURATION
# ============================================================================

# Animal keywords - if concept name contains these, filter it out
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

# Exclusion patterns (regex)
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

# Non-human semantic types (UMLS semantic types to exclude)
EXCLUDED_SEMANTIC_TYPES = {
    'Mammal',
    'Vertebrate',
    'Animal',
    'Bird',
    'Fish',
    'Amphibian',
    'Reptile',
    'Veterinary Medical Device',
    'Animal-Restricted Concept'
}

# Research/non-clinical semantic types to exclude
EXCLUDED_RESEARCH_TYPES = {
    'Research Activity',
    'Laboratory Procedure',
    'Experimental Model of Disease'
}


def is_animal_concept(concept_name, semantic_types=None):
    """
    Check if a concept is animal/veterinary related
    
    Args:
        concept_name: String name of the concept
        semantic_types: List of UMLS semantic types (optional)
    
    Returns:
        bool: True if concept should be filtered out
    """
    
    concept_lower = concept_name.lower()
    
    # Check for animal keywords
    for keyword in ANIMAL_KEYWORDS:
        if keyword in concept_lower:
            return True
    
    # Check for exclusion patterns
    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, concept_name, re.IGNORECASE):
            return True
    
    # Check semantic types if provided
    if semantic_types:
        for sem_type in semantic_types:
            if sem_type in EXCLUDED_SEMANTIC_TYPES:
                return True
            if sem_type in EXCLUDED_RESEARCH_TYPES:
                return True
    
    return False


def filter_concepts(concepts, keep_top_n=10, min_human_concepts=5):
    """
    Filter out animal/veterinary concepts from predictions
    
    Args:
        concepts: List of concept dicts with 'name', 'score', 'semantic_types', etc.
        keep_top_n: Maximum concepts to keep after filtering
        min_human_concepts: Minimum human concepts to return (keep filtering until we have this many)
    
    Returns:
        List of filtered concepts (human medicine only)
    """
    
    human_concepts = []
    filtered_concepts = []
    
    for concept in concepts:
        concept_name = concept.get('name', '')
        semantic_types = concept.get('semantic_types', [])
        
        if is_animal_concept(concept_name, semantic_types):
            filtered_concepts.append(concept['name'])
        else:
            human_concepts.append(concept)
            
            if len(human_concepts) >= keep_top_n:
                break
    
    # If we don't have enough human concepts, we might need to be less strict
    # But for now, return what we have
    if len(human_concepts) < min_human_concepts:
        print(f"  ⚠️  Warning: Only found {len(human_concepts)} human concepts (wanted {min_human_concepts})")
        if filtered_concepts:
            print(f"  🚫 Filtered out: {', '.join(filtered_concepts[:3])}...")
    
    return human_concepts


def get_filtered_top_concepts(concept_scores, concept_store, top_k=10):
    """
    Get top-K concepts with animal filtering applied
    
    Args:
        concept_scores: Numpy array of concept scores
        concept_store: Dict with concept information
        top_k: Number of human concepts to return
    
    Returns:
        List of filtered concept dicts
    """
    
    import numpy as np
    
    # Get all concepts sorted by score
    all_indices = np.argsort(concept_scores)[::-1]
    
    concepts = []
    filtered_count = 0
    
    for idx in all_indices:
        if len(concepts) >= top_k:
            break
        
        cui = concept_store['idx_to_concept'].get(idx, f'CUI_{idx}')
        concept_info = concept_store['concepts'].get(cui, {})
        
        concept_name = concept_info.get('preferred_name', f'Concept_{idx}')
        semantic_types = concept_info.get('semantic_types', [])
        
        # Check if animal concept
        if is_animal_concept(concept_name, semantic_types):
            filtered_count += 1
            continue
        
        concepts.append({
            'idx': idx,
            'cui': cui,
            'name': concept_name,
            'score': float(concept_scores[idx]),
            'semantic_types': semantic_types
        })
    
    if filtered_count > 0:
        print(f"  🚫 Filtered {filtered_count} animal/veterinary concepts")
    
    return concepts


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    """Test the filtering"""
    
    test_concepts = [
        {
            'name': 'Pneumonia, Atypical Interstitial, of Cattle',
            'score': 0.98,
            'semantic_types': ['Disease or Syndrome']
        },
        {
            'name': 'Pneumonia',
            'score': 0.95,
            'semantic_types': ['Disease or Syndrome']
        },
        {
            'name': 'Puerperal sepsis',
            'score': 0.94,
            'semantic_types': ['Disease or Syndrome']
        },
        {
            'name': 'African Swine Fever',
            'score': 0.93,
            'semantic_types': ['Disease or Syndrome']
        },
        {
            'name': 'Leukocytosis',
            'score': 0.92,
            'semantic_types': ['Finding']
        },
        {
            'name': 'Fever',
            'score': 0.91,
            'semantic_types': ['Sign or Symptom']
        }
    ]
    
    print("="*70)
    print("CONCEPT FILTER TEST")
    print("="*70)
    
    print("\n📋 Original concepts:")
    for i, c in enumerate(test_concepts, 1):
        print(f"  {i}. {c['name']} ({c['score']:.1%})")
    
    print("\n🔍 Filtering...")
    filtered = filter_concepts(test_concepts, keep_top_n=5)
    
    print(f"\n✅ Filtered concepts ({len(filtered)} remaining):")
    for i, c in enumerate(filtered, 1):
        print(f"  {i}. {c['name']} ({c['score']:.1%})")
    
    print("\n" + "="*70)
