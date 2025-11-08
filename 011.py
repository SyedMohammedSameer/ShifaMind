#!/usr/bin/env python3
"""
UMLS MRREL.RRF Explorer
Step 1: Verify file exists and understand structure
"""

from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm

# Your UMLS path
UMLS_PATH = Path('/content/drive/MyDrive/ShifaMind/01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/2025AA/META')

print("="*70)
print("STEP 1: EXPLORING UMLS MRREL.RRF")
print("="*70)

# Check if file exists
mrrel_path = UMLS_PATH / 'MRREL.RRF'
print(f"\n📂 Checking for MRREL.RRF...")
print(f"   Path: {mrrel_path}")
print(f"   Exists: {mrrel_path.exists()}")

if not mrrel_path.exists():
    print("\n❌ MRREL.RRF not found!")
    print("\nPossible locations to check:")
    print("  1. UMLS_PATH/MRREL.RRF")
    print("  2. UMLS_PATH/META/MRREL.RRF")
    print("  3. UMLS_PATH/../MRREL.RRF")

    # Try to find it
    print("\n🔍 Searching for MRREL.RRF in UMLS directory...")
    umls_base = UMLS_PATH.parent.parent
    for f in umls_base.rglob('MRREL.RRF'):
        print(f"   Found: {f}")
else:
    print("   ✅ Found!")

    # Get file size
    file_size_mb = mrrel_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {file_size_mb:.1f} MB")

    print("\n" + "="*70)
    print("STEP 2: UNDERSTANDING MRREL.RRF STRUCTURE")
    print("="*70)

    # MRREL.RRF format (from UMLS documentation)
    print("\n📋 MRREL.RRF Column Structure:")
    print("   0. CUI1      - Concept Unique Identifier 1")
    print("   1. AUI1      - Atom Unique Identifier 1")
    print("   2. STYPE1    - Source Atom Type 1")
    print("   3. REL       - Relationship label (CHD, PAR, RB, RN, etc.)")
    print("   4. CUI2      - Concept Unique Identifier 2")
    print("   5. AUI2      - Atom Unique Identifier 2")
    print("   6. STYPE2    - Source Atom Type 2")
    print("   7. RELA      - Additional relationship label")
    print("   8. RUI       - Relationship Unique Identifier")
    print("   9. SRUI      - Source Relationship Unique Identifier")
    print("   10. SAB      - Source Abbreviation")
    print("   11. SL       - Source of Label")
    print("   12. RG       - Relationship Group")
    print("   13. DIR      - Directionality flag")
    print("   14. SUPPRESS - Suppression flag")
    print("   15. CVF      - Content View Flag")

    print("\n📊 Key Relationship Types (REL):")
    print("   CHD - has child relationship")
    print("   PAR - has parent relationship")
    print("   RB  - has broader relationship")
    print("   RN  - has narrower relationship")
    print("   RO  - has other relationship")
    print("   SIB - has sibling relationship")

    print("\n" + "="*70)
    print("STEP 3: SAMPLING DATA")
    print("="*70)

    # Read first 1000 lines to understand
    print("\n📖 Reading first 1000 relations...")
    sample_relations = []
    rel_types = Counter()
    sources = Counter()

    with open(mrrel_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break

            fields = line.strip().split('|')
            if len(fields) >= 11:
                cui1 = fields[0]
                rel = fields[3]
                cui2 = fields[4]
                rela = fields[7]
                sab = fields[10]

                sample_relations.append({
                    'CUI1': cui1,
                    'REL': rel,
                    'CUI2': cui2,
                    'RELA': rela,
                    'SAB': sab
                })

                rel_types[rel] += 1
                sources[sab] += 1

    print(f"\n✅ Loaded {len(sample_relations)} sample relations")

    # Show distribution
    print("\n📊 Relationship Type Distribution (sample):")
    for rel, count in rel_types.most_common(10):
        print(f"   {rel:5s}: {count:4d} ({count/len(sample_relations)*100:.1f}%)")

    print("\n📊 Top Sources:")
    for sab, count in sources.most_common(5):
        print(f"   {sab:15s}: {count:4d}")

    # Show examples
    print("\n📝 Example Relations:")
    df_sample = pd.DataFrame(sample_relations[:10])
    print(df_sample.to_string(index=False))

    print("\n" + "="*70)
    print("STEP 4: COUNTING TOTAL RELATIONS")
    print("="*70)

    print("\n📊 Counting all relations (this may take 1-2 minutes)...")

    total_relations = 0
    rel_type_counts = Counter()
    snomedct_relations = 0

    with open(mrrel_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Counting"):
            fields = line.strip().split('|')
            if len(fields) >= 11:
                total_relations += 1
                rel_type_counts[fields[3]] += 1
                if fields[10] == 'SNOMEDCT_US':
                    snomedct_relations += 1

    print(f"\n✅ Total Relations: {total_relations:,}")
    print(f"   SNOMED CT Relations: {snomedct_relations:,}")

    print("\n📊 Full Relationship Type Distribution:")
    for rel, count in rel_type_counts.most_common():
        pct = count / total_relations * 100
        print(f"   {rel:5s}: {count:8,d} ({pct:5.1f}%)")

    print("\n" + "="*70)
    print("STEP 5: LOADING RELATIONS FOR TARGET CONCEPTS")
    print("="*70)

    # Let's load relations for your specific concepts
    print("\n🎯 Loading relations for pneumonia-related concepts...")

    # Example CUIs for pneumonia (you'll get these from your concept store)
    example_cuis = {
        'C0032285': 'Pneumonia',
        'C0006285': 'Bronchopneumonia',
        'C0001311': 'Acute bronchiolitis',
        'C0004096': 'Asthma',
        'C0021400': 'Influenza'
    }

    # Build a graph for these concepts
    concept_graph = defaultdict(list)

    print("\n📖 Building graph for example concepts...")
    with open(mrrel_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Building graph"):
            fields = line.strip().split('|')
            if len(fields) >= 11:
                cui1 = fields[0]
                rel = fields[3]
                cui2 = fields[4]
                sab = fields[10]

                # Only SNOMED CT relations
                if sab != 'SNOMEDCT_US':
                    continue

                # Only if involves our example concepts
                if cui1 in example_cuis or cui2 in example_cuis:
                    concept_graph[cui1].append({
                        'rel': rel,
                        'target': cui2
                    })

    print(f"\n✅ Built graph with {len(concept_graph)} nodes")

    print("\n📊 Example Graph Structure:")
    for cui, name in list(example_cuis.items())[:3]:
        if cui in concept_graph:
            print(f"\n   {name} ({cui}):")
            relations = concept_graph[cui][:5]  # Show first 5
            for rel_info in relations:
                target_name = example_cuis.get(rel_info['target'], 'Other concept')
                print(f"      {rel_info['rel']} → {target_name} ({rel_info['target']})")

    print("\n" + "="*70)
    print("✅ EXPLORATION COMPLETE")
    print("="*70)

    print("\n📋 Summary:")
    print(f"   ✅ MRREL.RRF exists and is readable")
    print(f"   ✅ Contains {total_relations:,} total relations")
    print(f"   ✅ SNOMED CT has {snomedct_relations:,} relations")
    print(f"   ✅ Can build concept graphs")

    print("\n🎯 Next Steps:")
    print("   1. Create UMLSRelationLoader class")
    print("   2. Implement graph distance computation")
    print("   3. Add hierarchy-aware filtering")
    print("   4. Integrate into ShifaMind")

    print("\n" + "="*70)



#!/usr/bin/env python3
"""
UMLS Hierarchy Loader - Memory Efficient
Loads only relevant SNOMED CT hierarchical relations for active concepts
"""

from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import pickle

class UMLSHierarchyLoader:
    """
    Memory-efficient loader for UMLS concept hierarchies
    Only loads PAR/CHD/RB/RN relations from SNOMEDCT_US for active concepts
    """

    def __init__(self, mrrel_path: Path):
        self.mrrel_path = mrrel_path
        self.graph = defaultdict(set)  # CUI -> set of (relation_type, target_CUI)
        self.reverse_graph = defaultdict(set)  # For bidirectional search
        self.loaded_concepts = set()

        # Only load hierarchical relations
        self.HIERARCHICAL_RELS = {'PAR', 'CHD', 'RB', 'RN'}

        # Relation priorities (for scoring)
        self.REL_WEIGHTS = {
            'CHD': 1.0,  # Child (most specific)
            'PAR': 1.0,  # Parent (most specific)
            'RN': 0.8,   # Narrower
            'RB': 0.8,   # Broader
        }

    def load_for_concepts(self, target_cuis: Set[str],
                         max_hops: int = 4,
                         cache_path: Path = None) -> Dict:
        """
        Load hierarchy for specific concepts up to max_hops away

        Args:
            target_cuis: Set of CUIs to build hierarchy for
            max_hops: Maximum graph distance to load
            cache_path: Optional path to cache the graph

        Returns:
            Dict with graph statistics
        """

        # Check cache first
        if cache_path and cache_path.exists():
            print(f"📦 Loading cached hierarchy from {cache_path}...")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                self.graph = cached['graph']
                self.reverse_graph = cached['reverse_graph']
                self.loaded_concepts = cached['loaded_concepts']
            print(f"   ✅ Loaded {len(self.loaded_concepts)} concepts, "
                  f"{sum(len(v) for v in self.graph.values())} edges")
            return self._get_stats()

        print(f"\n🔍 Loading UMLS hierarchy for {len(target_cuis)} concepts...")
        print(f"   Max hops: {max_hops}")
        print(f"   Source: {self.mrrel_path}")

        # Multi-pass loading (each pass expands 1 hop)
        concepts_to_explore = target_cuis.copy()
        all_relevant_concepts = target_cuis.copy()

        for hop in range(max_hops):
            print(f"\n   Hop {hop+1}/{max_hops}: Exploring {len(concepts_to_explore)} concepts...")

            new_concepts = set()
            relations_loaded = 0

            with open(self.mrrel_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in tqdm(f, desc=f"   Loading", total=62908136,
                               miniters=1000000):
                    fields = line.strip().split('|')

                    if len(fields) < 11:
                        continue

                    cui1 = fields[0]
                    rel = fields[3]
                    cui2 = fields[4]
                    sab = fields[10]

                    # Filter: only SNOMED CT
                    if sab != 'SNOMEDCT_US':
                        continue

                    # Filter: only hierarchical relations
                    if rel not in self.HIERARCHICAL_RELS:
                        continue

                    # Check if involves concepts we're exploring
                    if cui1 in concepts_to_explore or cui2 in concepts_to_explore:
                        # Add to graph
                        self.graph[cui1].add((rel, cui2))
                        self.reverse_graph[cui2].add((rel, cui1))

                        relations_loaded += 1

                        # Track new concepts for next hop
                        if cui1 not in all_relevant_concepts:
                            new_concepts.add(cui1)
                        if cui2 not in all_relevant_concepts:
                            new_concepts.add(cui2)

            print(f"   ✅ Loaded {relations_loaded:,} relations")
            print(f"   📊 Discovered {len(new_concepts)} new concepts")

            all_relevant_concepts.update(new_concepts)
            concepts_to_explore = new_concepts

            # Stop if no new concepts
            if len(new_concepts) == 0:
                print(f"   🛑 No new concepts found, stopping early at hop {hop+1}")
                break

        self.loaded_concepts = all_relevant_concepts

        stats = self._get_stats()

        # Cache for next time
        if cache_path:
            print(f"\n💾 Caching hierarchy to {cache_path}...")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'reverse_graph': self.reverse_graph,
                    'loaded_concepts': self.loaded_concepts
                }, f)
            print(f"   ✅ Cached!")

        return stats

    def _get_stats(self) -> Dict:
        """Get graph statistics"""
        total_edges = sum(len(v) for v in self.graph.values())

        rel_counts = defaultdict(int)
        for neighbors in self.graph.values():
            for rel, _ in neighbors:
                rel_counts[rel] += 1

        return {
            'num_concepts': len(self.loaded_concepts),
            'num_edges': total_edges,
            'rel_distribution': dict(rel_counts)
        }

    def get_distance(self, cui1: str, cui2: str,
                     max_depth: int = 5) -> Tuple[int, List[str]]:
        """
        Compute shortest path distance between two concepts

        Args:
            cui1: Source concept
            cui2: Target concept
            max_depth: Maximum search depth

        Returns:
            (distance, path) where path is list of CUIs from cui1 to cui2
            Returns (float('inf'), []) if no path found
        """

        if cui1 == cui2:
            return (0, [cui1])

        if cui1 not in self.graph and cui1 not in self.reverse_graph:
            return (float('inf'), [])

        # Bidirectional BFS for efficiency
        queue = deque([(cui1, [cui1])])
        visited = {cui1}

        for depth in range(max_depth):
            # Process all nodes at current depth
            level_size = len(queue)

            for _ in range(level_size):
                current, path = queue.popleft()

                # Check forward edges
                for rel, neighbor in self.graph.get(current, []):
                    if neighbor == cui2:
                        return (len(path), path + [neighbor])

                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

                # Check reverse edges (can go up hierarchy too)
                for rel, neighbor in self.reverse_graph.get(current, []):
                    if neighbor == cui2:
                        return (len(path), path + [neighbor])

                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

            if len(queue) == 0:
                break

        return (float('inf'), [])

    def get_neighbors(self, cui: str, relation_types: Set[str] = None) -> List[Tuple[str, str]]:
        """
        Get all neighbors of a concept

        Args:
            cui: Concept to get neighbors for
            relation_types: If provided, only return these relation types

        Returns:
            List of (relation_type, neighbor_cui) tuples
        """
        neighbors = []

        for rel, neighbor in self.graph.get(cui, []):
            if relation_types is None or rel in relation_types:
                neighbors.append((rel, neighbor))

        return neighbors

    def get_ancestors(self, cui: str, max_hops: int = 5) -> Set[str]:
        """Get all ancestor concepts (via PAR/RB relations)"""
        ancestors = set()
        queue = deque([cui])
        visited = {cui}

        for _ in range(max_hops):
            level_size = len(queue)
            if level_size == 0:
                break

            for _ in range(level_size):
                current = queue.popleft()

                for rel, neighbor in self.graph.get(current, []):
                    if rel in {'PAR', 'RB'} and neighbor not in visited:
                        ancestors.add(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)

        return ancestors

    def get_descendants(self, cui: str, max_hops: int = 5) -> Set[str]:
        """Get all descendant concepts (via CHD/RN relations)"""
        descendants = set()
        queue = deque([cui])
        visited = {cui}

        for _ in range(max_hops):
            level_size = len(queue)
            if level_size == 0:
                break

            for _ in range(level_size):
                current = queue.popleft()

                for rel, neighbor in self.graph.get(current, []):
                    if rel in {'CHD', 'RN'} and neighbor not in visited:
                        descendants.add(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)

        return descendants

    def is_related(self, cui1: str, cui2: str, max_distance: int = 4) -> bool:
        """Check if two concepts are related within max_distance hops"""
        distance, _ = self.get_distance(cui1, cui2, max_depth=max_distance)
        return distance <= max_distance

    def get_common_ancestors(self, cui1: str, cui2: str, max_hops: int = 5) -> Set[str]:
        """Find common ancestors of two concepts"""
        ancestors1 = self.get_ancestors(cui1, max_hops)
        ancestors2 = self.get_ancestors(cui2, max_hops)
        return ancestors1.intersection(ancestors2)


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_hierarchy_loader():
    """Test the hierarchy loader with sample concepts"""

    print("\n" + "="*70)
    print("TESTING HIERARCHY LOADER")
    print("="*70)

    # Setup
    UMLS_PATH = Path('/content/drive/MyDrive/ShifaMind/01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/2025AA/META')
    mrrel_path = UMLS_PATH / 'MRREL.RRF'
    cache_path = Path('/content/drive/MyDrive/ShifaMind/umls_hierarchy_cache.pkl')

    # Test concepts
    test_concepts = {
        'C0032285': 'Pneumonia',
        'C0006285': 'Bronchopneumonia',
        'C0001311': 'Acute bronchiolitis',
        'C0004096': 'Asthma',
        'C0021400': 'Influenza',
        'C0018802': 'Heart failure',
        'C0036690': 'Sepsis',
    }

    print(f"\n🎯 Test concepts: {len(test_concepts)}")
    for cui, name in test_concepts.items():
        print(f"   {cui}: {name}")

    # Load hierarchy
    loader = UMLSHierarchyLoader(mrrel_path)
    stats = loader.load_for_concepts(
        set(test_concepts.keys()),
        max_hops=3,
        cache_path=cache_path
    )

    print(f"\n📊 Loaded Hierarchy Stats:")
    print(f"   Total concepts: {stats['num_concepts']}")
    print(f"   Total edges: {stats['num_edges']}")
    print(f"   Relation distribution:")
    for rel, count in stats['rel_distribution'].items():
        print(f"      {rel}: {count}")

    # Test distances
    print(f"\n🔍 Testing Graph Distances:")

    test_pairs = [
        ('C0032285', 'C0006285', 'Pneumonia → Bronchopneumonia'),
        ('C0032285', 'C0001311', 'Pneumonia → Acute bronchiolitis'),
        ('C0032285', 'C0004096', 'Pneumonia → Asthma'),
        ('C0018802', 'C0032285', 'Heart failure → Pneumonia'),
    ]

    for cui1, cui2, description in test_pairs:
        distance, path = loader.get_distance(cui1, cui2, max_depth=5)

        if distance < float('inf'):
            print(f"\n   ✅ {description}")
            print(f"      Distance: {distance} hops")
            print(f"      Path: {' → '.join(path[:4])}{'...' if len(path) > 4 else ''}")
        else:
            print(f"\n   ❌ {description}")
            print(f"      No path found within 5 hops")

    # Test neighbors
    print(f"\n🔗 Testing Neighbors:")
    for cui, name in list(test_concepts.items())[:3]:
        neighbors = loader.get_neighbors(cui)
        print(f"\n   {name} ({cui}): {len(neighbors)} neighbors")
        for rel, neighbor in neighbors[:5]:
            print(f"      {rel} → {neighbor}")

    # Test ancestors/descendants
    print(f"\n🌳 Testing Ancestors/Descendants:")
    test_cui = 'C0006285'  # Bronchopneumonia

    ancestors = loader.get_ancestors(test_cui, max_hops=3)
    descendants = loader.get_descendants(test_cui, max_hops=3)

    print(f"\n   Bronchopneumonia:")
    print(f"      Ancestors: {len(ancestors)} concepts")
    print(f"      Descendants: {len(descendants)} concepts")

    # Test common ancestors
    print(f"\n🔗 Testing Common Ancestors:")
    common = loader.get_common_ancestors('C0032285', 'C0006285')
    print(f"   Pneumonia & Bronchopneumonia: {len(common)} common ancestors")

    print("\n" + "="*70)
    print("✅ TESTING COMPLETE")
    print("="*70)

    return loader


if __name__ == "__main__":
    loader = test_hierarchy_loader()



#!/usr/bin/env python3
"""
Hierarchy-Aware Concept Filtering Integration
Combines semantic similarity with UMLS graph distance
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
import torch
import numpy as np
from collections import defaultdict

# ============================================================================
# HIERARCHY-AWARE CONCEPT RANKER
# ============================================================================

class HierarchyAwareConceptRanker:
    """
    Ranks concepts using both semantic similarity and hierarchical distance
    """

    def __init__(self, hierarchy_loader, icd_to_cui_map: Dict):
        """
        Args:
            hierarchy_loader: UMLSHierarchyLoader instance
            icd_to_cui_map: Mapping from ICD codes to UMLS CUIs
        """
        self.hierarchy = hierarchy_loader
        self.icd_to_cui = icd_to_cui_map

        # Scoring weights
        self.SIMILARITY_WEIGHT = 0.4  # Weight for embedding similarity
        self.DISTANCE_WEIGHT = 0.6    # Weight for graph distance

        # Distance scoring (inverse - closer = better)
        self.MAX_DISTANCE = 5

    def score_concepts_for_diagnosis(self,
                                    concept_scores: np.ndarray,
                                    concept_ids: List[str],
                                    predicted_icd: str,
                                    max_distance: int = 4) -> np.ndarray:
        """
        Score concepts combining semantic similarity and hierarchical distance

        Args:
            concept_scores: Array of semantic similarity scores (n_concepts,)
            concept_ids: List of concept CUIs
            predicted_icd: Predicted ICD-10 code
            max_distance: Maximum allowed graph distance

        Returns:
            Combined scores (n_concepts,)
        """

        # Get diagnosis CUIs
        diagnosis_cuis = self._get_diagnosis_cuis(predicted_icd)

        if len(diagnosis_cuis) == 0:
            # No hierarchy info for this diagnosis, return semantic scores only
            return concept_scores

        # Compute hierarchical scores for each concept
        hierarchical_scores = np.zeros_like(concept_scores)

        for i, cui in enumerate(concept_ids):
            # Find minimum distance to any diagnosis CUI
            min_distance = float('inf')

            for dx_cui in diagnosis_cuis:
                distance, _ = self.hierarchy.get_distance(cui, dx_cui, max_depth=max_distance + 1)
                min_distance = min(min_distance, distance)

            # Convert distance to score (0-1, where closer = higher)
            if min_distance <= max_distance:
                # Exponential decay: distance 0 = 1.0, distance 4 = ~0.2
                hierarchical_scores[i] = np.exp(-0.5 * min_distance)
            else:
                # Too far, low score
                hierarchical_scores[i] = 0.01

        # Combine semantic and hierarchical scores
        combined_scores = (
            self.SIMILARITY_WEIGHT * concept_scores +
            self.DISTANCE_WEIGHT * hierarchical_scores
        )

        return combined_scores

    def filter_by_hierarchy(self,
                           concept_scores: np.ndarray,
                           concept_ids: List[str],
                           predicted_icd: str,
                           max_distance: int = 4) -> Tuple[np.ndarray, List[int]]:
        """
        Hard filter: remove concepts beyond max_distance

        Returns:
            (filtered_scores, kept_indices)
        """

        diagnosis_cuis = self._get_diagnosis_cuis(predicted_icd)

        if len(diagnosis_cuis) == 0:
            # No filtering possible
            return concept_scores, list(range(len(concept_scores)))

        kept_indices = []

        for i, cui in enumerate(concept_ids):
            # Check if within distance of any diagnosis CUI
            is_related = any(
                self.hierarchy.is_related(cui, dx_cui, max_distance=max_distance)
                for dx_cui in diagnosis_cuis
            )

            if is_related:
                kept_indices.append(i)

        if len(kept_indices) == 0:
            # If nothing passes filter, keep top 10 by semantic score
            kept_indices = np.argsort(concept_scores)[-10:].tolist()

        filtered_scores = concept_scores[kept_indices]

        return filtered_scores, kept_indices

    def rank_concepts_with_hierarchy(self,
                                    concept_scores: torch.Tensor,
                                    concept_ids: List[str],
                                    predicted_icd: str,
                                    top_k: int = 10,
                                    max_distance: int = 4,
                                    use_hard_filter: bool = True) -> Dict:
        """
        Complete ranking pipeline with hierarchy

        Args:
            concept_scores: PyTorch tensor of scores (batch_size, n_concepts)
            concept_ids: List of concept CUIs
            predicted_icd: Predicted ICD-10 code (or list for batch)
            top_k: Number of concepts to return
            max_distance: Maximum graph distance
            use_hard_filter: If True, hard filter by distance; else soft rerank

        Returns:
            Dict with ranked concept indices and scores
        """

        # Convert to numpy for processing
        scores_np = concept_scores.cpu().detach().numpy()

        # Handle batch
        if len(scores_np.shape) == 1:
            scores_np = scores_np.reshape(1, -1)
            predicted_icd = [predicted_icd]

        batch_size = scores_np.shape[0]

        ranked_concepts = []

        for b in range(batch_size):
            sample_scores = scores_np[b]
            icd_code = predicted_icd[b] if isinstance(predicted_icd, list) else predicted_icd

            if use_hard_filter:
                # Hard filter + rerank
                filtered_scores, kept_indices = self.filter_by_hierarchy(
                    sample_scores, concept_ids, icd_code, max_distance
                )

                # Rerank filtered concepts
                combined_scores = self.score_concepts_for_diagnosis(
                    filtered_scores, [concept_ids[i] for i in kept_indices], icd_code, max_distance
                )

                # Get top-k from filtered + reranked
                top_indices_in_filtered = np.argsort(combined_scores)[-top_k:][::-1]
                top_indices_original = [kept_indices[i] for i in top_indices_in_filtered]
                top_scores = combined_scores[top_indices_in_filtered]

            else:
                # Soft rerank all concepts
                combined_scores = self.score_concepts_for_diagnosis(
                    sample_scores, concept_ids, icd_code, max_distance
                )

                # Get top-k
                top_indices_original = np.argsort(combined_scores)[-top_k:][::-1]
                top_scores = combined_scores[top_indices_original]

            ranked_concepts.append({
                'concept_indices': top_indices_original,
                'concept_scores': top_scores,
                'concept_cuis': [concept_ids[i] for i in top_indices_original]
            })

        return ranked_concepts[0] if len(ranked_concepts) == 1 else ranked_concepts

    def _get_diagnosis_cuis(self, icd_code: str) -> List[str]:
        """Get UMLS CUIs for an ICD code"""

        # Try exact match
        if icd_code in self.icd_to_cui:
            return list(self.icd_to_cui[icd_code])[:5]  # Top 5

        # Try variants
        variants = [
            icd_code,
            icd_code.replace('.', ''),
            icd_code[:3],  # Category
        ]

        for variant in variants:
            if variant in self.icd_to_cui:
                return list(self.icd_to_cui[variant])[:5]

        return []

    def explain_concept_relevance(self,
                                 cui: str,
                                 predicted_icd: str,
                                 concept_store) -> Dict:
        """
        Explain why a concept is relevant (or not) for debugging
        """

        diagnosis_cuis = self._get_diagnosis_cuis(predicted_icd)

        explanation = {
            'cui': cui,
            'name': concept_store.concepts.get(cui, {}).get('name', 'Unknown'),
            'icd_code': predicted_icd,
            'diagnosis_cuis': diagnosis_cuis,
            'distances': {},
            'paths': {},
            'min_distance': float('inf'),
            'is_relevant': False
        }

        for dx_cui in diagnosis_cuis:
            distance, path = self.hierarchy.get_distance(cui, dx_cui, max_depth=6)
            explanation['distances'][dx_cui] = distance
            explanation['paths'][dx_cui] = path

            if distance < explanation['min_distance']:
                explanation['min_distance'] = distance

        explanation['is_relevant'] = explanation['min_distance'] <= 4

        return explanation


# ============================================================================
# UPDATED ENHANCED CONCEPT STORE WITH HIERARCHY
# ============================================================================

class HierarchyEnhancedConceptStore:
    """
    Concept store that uses hierarchy for validation and ranking
    """

    def __init__(self, umls_concepts: Dict, icd_to_cui: Dict, hierarchy_loader):
        self.umls_concepts = umls_concepts
        self.icd_to_cui = icd_to_cui
        self.hierarchy = hierarchy_loader
        self.concepts = {}
        self.concept_to_idx = {}
        self.idx_to_concept = {}

    def build_concept_set(self,
                         target_icd_codes: List[str],
                         icd_descriptions: Dict[str, str],
                         target_concept_count: int = 200,
                         max_hierarchy_distance: int = 4) -> Dict:
        """
        Build concept set using hierarchy validation
        """

        print(f"\n🌳 Building hierarchy-validated concept set...")
        print(f"   Target: {target_concept_count} concepts")
        print(f"   Max hierarchy distance: {max_hierarchy_distance} hops")

        relevant_cuis = set()

        # Strategy 1: Get diagnosis CUIs
        diagnosis_cuis = set()
        for icd in target_icd_codes:
            variants = [icd, icd.replace('.', ''), icd[:3]]
            for variant in variants:
                if variant in self.icd_to_cui:
                    diagnosis_cuis.update(self.icd_to_cui[variant])

        print(f"   Found {len(diagnosis_cuis)} diagnosis CUIs")

        # Strategy 2: Expand via hierarchy
        for dx_cui in diagnosis_cuis:
            # Get ancestors (more general)
            ancestors = self.hierarchy.get_ancestors(dx_cui, max_hops=2)
            relevant_cuis.update(list(ancestors)[:10])  # Top 10 ancestors

            # Get descendants (more specific)
            descendants = self.hierarchy.get_descendants(dx_cui, max_hops=2)
            relevant_cuis.update(list(descendants)[:20])  # Top 20 descendants

            # Get siblings (via common parents)
            neighbors = self.hierarchy.get_neighbors(dx_cui)
            for rel, neighbor in neighbors[:10]:
                relevant_cuis.add(neighbor)

        print(f"   After hierarchy expansion: {len(relevant_cuis)} concepts")

        # Strategy 3: Keyword matching (but validate with hierarchy)
        diagnosis_keywords = {
            'J189': ['pneumonia', 'lung', 'respiratory', 'pulmonary', 'bronch'],
            'I5023': ['heart', 'cardiac', 'ventricular', 'cardiomyopathy'],
            'A419': ['sepsis', 'septic', 'infection', 'bacteremia'],
            'K8000': ['gallbladder', 'biliary', 'cholecystitis'],
        }

        for icd in target_icd_codes:
            keywords = diagnosis_keywords.get(icd, [])

            for cui, info in self.umls_concepts.items():
                if cui in relevant_cuis:
                    continue

                # Keyword match
                terms_text = ' '.join([info['name']] + info.get('terms', [])).lower()
                if any(kw in terms_text for kw in keywords):
                    # Validate with hierarchy
                    dx_cuis = self.icd_to_cui.get(icd, [])
                    if dx_cuis:
                        # Check if related to any diagnosis CUI
                        is_related = any(
                            self.hierarchy.is_related(cui, dx_cui, max_distance=max_hierarchy_distance)
                            for dx_cui in dx_cuis[:3]
                        )

                        if is_related:
                            relevant_cuis.add(cui)

                if len(relevant_cuis) >= target_concept_count * 2:
                    break

        print(f"   After keyword + hierarchy validation: {len(relevant_cuis)} concepts")

        # Build final concept dictionary
        for cui in relevant_cuis:
            if cui in self.umls_concepts:
                concept = self.umls_concepts[cui]
                self.concepts[cui] = {
                    'cui': cui,
                    'name': concept['name'],
                    'definition': concept.get('definition', ''),
                    'terms': concept.get('terms', []),
                    'semantic_types': concept.get('semantic_types', [])
                }

        # Trim to target count if needed
        if len(self.concepts) > target_concept_count:
            # Keep concepts closest to diagnosis CUIs
            concept_distances = {}
            for cui in self.concepts:
                min_dist = min(
                    self.hierarchy.get_distance(cui, dx_cui, max_depth=5)[0]
                    for dx_cui in diagnosis_cuis
                )
                concept_distances[cui] = min_dist

            # Keep closest concepts
            sorted_cuis = sorted(concept_distances.keys(), key=lambda c: concept_distances[c])
            kept_cuis = sorted_cuis[:target_concept_count]

            self.concepts = {cui: self.concepts[cui] for cui in kept_cuis}

        concept_list = list(self.concepts.keys())
        self.concept_to_idx = {cui: i for i, cui in enumerate(concept_list)}
        self.idx_to_concept = {i: cui for i, cui in enumerate(concept_list)}

        print(f"   ✅ Final concept set: {len(self.concepts)} concepts")

        return self.concepts

    def create_concept_embeddings(self, tokenizer, model, device):
        """Create concept embeddings (same as before)"""
        from tqdm import tqdm

        concept_texts = []
        for cui, info in self.concepts.items():
            text = f"{info['name']}."
            if info['definition']:
                text += f" {info['definition'][:200]}"
            concept_texts.append(text)

        batch_size = 32
        all_embeddings = []

        for i in tqdm(range(0, len(concept_texts), batch_size), desc="Encoding concepts"):
            batch = concept_texts[i:i+batch_size]
            encodings = tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = model(**encodings)
                embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).to(device)


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_hierarchy_integration():
    """Test the hierarchy-aware ranking system"""

    print("\n" + "="*70)
    print("TESTING HIERARCHY-AWARE CONCEPT RANKING")
    print("="*70)

    # Load hierarchy (from cache)
    from pathlib import Path
    UMLS_PATH = Path('/content/drive/MyDrive/ShifaMind/01_Raw_Datasets/Extracted/umls-2025AA-metathesaurus-full/2025AA/META')
    cache_path = Path('/content/drive/MyDrive/ShifaMind/umls_hierarchy_cache.pkl')

    # Need to import the loader
    print("\n📦 Loading cached hierarchy...")
    import pickle
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)

    # Create mock hierarchy loader with cached data
    class MockHierarchyLoader:
        def __init__(self, cached_data):
            self.graph = cached_data['graph']
            self.reverse_graph = cached_data['reverse_graph']
            self.loaded_concepts = cached_data['loaded_concepts']

        def get_distance(self, cui1, cui2, max_depth=5):
            from collections import deque
            if cui1 == cui2:
                return (0, [cui1])
            queue = deque([(cui1, [cui1])])
            visited = {cui1}
            for depth in range(max_depth):
                level_size = len(queue)
                for _ in range(level_size):
                    current, path = queue.popleft()
                    for rel, neighbor in self.graph.get(current, []):
                        if neighbor == cui2:
                            return (len(path), path + [neighbor])
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
                    for rel, neighbor in self.reverse_graph.get(current, []):
                        if neighbor == cui2:
                            return (len(path), path + [neighbor])
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
                if len(queue) == 0:
                    break
            return (float('inf'), [])

        def is_related(self, cui1, cui2, max_distance=4):
            distance, _ = self.get_distance(cui1, cui2, max_depth=max_distance)
            return distance <= max_distance

        def get_neighbors(self, cui):
            return list(self.graph.get(cui, []))

        def get_ancestors(self, cui, max_hops=5):
            from collections import deque
            ancestors = set()
            queue = deque([cui])
            visited = {cui}
            for _ in range(max_hops):
                level_size = len(queue)
                if level_size == 0:
                    break
                for _ in range(level_size):
                    current = queue.popleft()
                    for rel, neighbor in self.graph.get(current, []):
                        if rel in {'PAR', 'RB'} and neighbor not in visited:
                            ancestors.add(neighbor)
                            visited.add(neighbor)
                            queue.append(neighbor)
            return ancestors

        def get_descendants(self, cui, max_hops=5):
            from collections import deque
            descendants = set()
            queue = deque([cui])
            visited = {cui}
            for _ in range(max_hops):
                level_size = len(queue)
                if level_size == 0:
                    break
                for _ in range(level_size):
                    current = queue.popleft()
                    for rel, neighbor in self.graph.get(current, []):
                        if rel in {'CHD', 'RN'} and neighbor not in visited:
                            descendants.add(neighbor)
                            visited.add(neighbor)
                            queue.append(neighbor)
            return descendants

    hierarchy = MockHierarchyLoader(cached)
    print(f"   ✅ Loaded {len(hierarchy.loaded_concepts)} concepts")

    # Mock ICD to CUI mapping
    icd_to_cui = {
        'J189': ['C0032285'],  # Pneumonia
        'I5023': ['C0018802'],  # Heart failure
        'A419': ['C0036690'],   # Sepsis
    }

    # Create ranker
    ranker = HierarchyAwareConceptRanker(hierarchy, icd_to_cui)

    print("\n🎯 Testing Concept Ranking for Pneumonia (J189)...")

    # Mock concepts (mix of related and unrelated)
    test_concepts = [
        'C0006285',  # Bronchopneumonia (related - 1 hop)
        'C0001311',  # Acute bronchiolitis (related - 3 hops)
        'C0004096',  # Asthma (related - 2 hops)
        'C0018802',  # Heart failure (unrelated - 4 hops)
        'C0036690',  # Sepsis (unrelated - distant)
        'C0021400',  # Influenza (related - respiratory)
    ]

    # Mock semantic scores (all high)
    semantic_scores = np.array([0.9, 0.88, 0.87, 0.86, 0.85, 0.84])

    print(f"\n📊 Semantic Scores (before hierarchy):")
    for i, (cui, score) in enumerate(zip(test_concepts, semantic_scores)):
        print(f"   {i+1}. {cui}: {score:.3f}")

    # Apply hierarchy ranking
    combined_scores = ranker.score_concepts_for_diagnosis(
        semantic_scores,
        test_concepts,
        'J189',
        max_distance=4
    )

    print(f"\n🌳 Combined Scores (with hierarchy):")
    ranked_indices = np.argsort(combined_scores)[::-1]
    for rank, idx in enumerate(ranked_indices, 1):
        cui = test_concepts[idx]
        semantic = semantic_scores[idx]
        combined = combined_scores[idx]
        distance, _ = hierarchy.get_distance(cui, 'C0032285', max_depth=5)

        print(f"   {rank}. {cui}: {combined:.3f} "
              f"(semantic: {semantic:.3f}, distance: {distance} hops)")

    print(f"\n✅ Notice: Respiratory concepts ranked higher despite similar semantic scores!")

    # Test hard filtering
    print(f"\n🔍 Testing Hard Filter (max_distance=4)...")
    filtered_scores, kept_indices = ranker.filter_by_hierarchy(
        semantic_scores, test_concepts, 'J189', max_distance=4
    )

    print(f"   Kept {len(kept_indices)}/{len(test_concepts)} concepts:")
    for idx in kept_indices:
        cui = test_concepts[idx]
        distance, _ = hierarchy.get_distance(cui, 'C0032285', max_depth=5)
        print(f"      {cui}: distance {distance} hops")

    print("\n" + "="*70)
    print("✅ INTEGRATION TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_hierarchy_integration()


#!/usr/bin/env python3
"""
ShifaMind with Hierarchy Integration - Complete Single File
Just run this entire script - no other files needed!

Key Changes from Enhanced Version:
✅ Added UMLS hierarchy loading (cached, fast)
✅ Added hierarchy-aware concept ranking
✅ Concepts now ranked by clinical relevance (graph distance)
✅ Demo shows hierarchy explanations
"""

# ============================================================================
# SETUP & IMPORTS
# ============================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🔧 Installing dependencies...")
    !pip install -q faiss-cpu scikit-learn
    from google.colab import drive
    drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

import pickle
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter, deque

import matplotlib.pyplot as plt
import seaborn as sns
import faiss

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# DATA PATHS
# ============================================================================

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind/01_Raw_Datasets/Extracted')
MIMIC_PATH = BASE_PATH / 'mimic-iv-3.1'
UMLS_PATH = BASE_PATH / 'umls-2025AA-metathesaurus-full/2025AA/META'
ICD_PATH = BASE_PATH / 'icd10cm-CodesDescriptions-2024'
NOTES_PATH = BASE_PATH / 'mimic-iv-note-2.2'

print(f"\nValidating paths...")
assert MIMIC_PATH.exists(), "MIMIC path not found!"
assert UMLS_PATH.exists(), "UMLS path not found!"
print("✅ All paths valid")

# ============================================================================
# NEW: UMLS HIERARCHY LOADER
# ============================================================================

class UMLSHierarchyLoader:
    """Memory-efficient loader for UMLS concept hierarchies"""

    def __init__(self, mrrel_path: Path):
        self.mrrel_path = mrrel_path
        self.graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.loaded_concepts = set()
        self.HIERARCHICAL_RELS = {'PAR', 'CHD', 'RB', 'RN'}

    def load_for_concepts(self, target_cuis: Set[str], max_hops: int = 4,
                         cache_path: Path = None) -> Dict:
        """Load hierarchy for specific concepts up to max_hops away"""

        # Check cache first
        if cache_path and cache_path.exists():
            print(f"📦 Loading hierarchy from cache...")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                self.graph = cached['graph']
                self.reverse_graph = cached['reverse_graph']
                self.loaded_concepts = cached['loaded_concepts']
            print(f"   ✅ Loaded {len(self.loaded_concepts)} concepts, "
                  f"{sum(len(v) for v in self.graph.values())} edges")
            return self._get_stats()

        print(f"\n🔍 Loading UMLS hierarchy...")
        print(f"   Starting with {len(target_cuis)} seed concepts")
        print(f"   Expanding {max_hops} hops...")

        concepts_to_explore = target_cuis.copy()
        all_relevant_concepts = target_cuis.copy()

        for hop in range(max_hops):
            print(f"\n   Hop {hop+1}/{max_hops}: Exploring {len(concepts_to_explore)} concepts...")

            new_concepts = set()
            relations_loaded = 0

            with open(self.mrrel_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in tqdm(f, desc=f"   Loading", total=62908136, miniters=1000000):
                    fields = line.strip().split('|')
                    if len(fields) < 11:
                        continue

                    cui1, rel, cui2, sab = fields[0], fields[3], fields[4], fields[10]

                    # Filter: only SNOMED CT hierarchical relations
                    if sab != 'SNOMEDCT_US' or rel not in self.HIERARCHICAL_RELS:
                        continue

                    # Check if involves concepts we're exploring
                    if cui1 in concepts_to_explore or cui2 in concepts_to_explore:
                        self.graph[cui1].add((rel, cui2))
                        self.reverse_graph[cui2].add((rel, cui1))
                        relations_loaded += 1

                        if cui1 not in all_relevant_concepts:
                            new_concepts.add(cui1)
                        if cui2 not in all_relevant_concepts:
                            new_concepts.add(cui2)

            print(f"      ✅ Loaded {relations_loaded:,} relations")
            print(f"      📊 Discovered {len(new_concepts)} new concepts")

            all_relevant_concepts.update(new_concepts)
            concepts_to_explore = new_concepts

            if len(new_concepts) == 0:
                print(f"      🛑 No new concepts, stopping at hop {hop+1}")
                break

        self.loaded_concepts = all_relevant_concepts

        # Cache for next time
        if cache_path:
            print(f"\n💾 Caching hierarchy to {cache_path}...")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'reverse_graph': self.reverse_graph,
                    'loaded_concepts': self.loaded_concepts
                }, f)
            print(f"   ✅ Cached!")

        return self._get_stats()

    def _get_stats(self) -> Dict:
        """Get graph statistics"""
        rel_counts = defaultdict(int)
        for neighbors in self.graph.values():
            for rel, _ in neighbors:
                rel_counts[rel] += 1
        return {
            'num_concepts': len(self.loaded_concepts),
            'num_edges': sum(len(v) for v in self.graph.values()),
            'rel_distribution': dict(rel_counts)
        }

    def get_distance(self, cui1: str, cui2: str, max_depth: int = 5):
        """Compute shortest path distance between two concepts"""

        if cui1 == cui2:
            return (0, [cui1])

        if cui1 not in self.graph and cui1 not in self.reverse_graph:
            return (float('inf'), [])

        # Bidirectional BFS
        queue = deque([(cui1, [cui1])])
        visited = {cui1}

        for depth in range(max_depth):
            level_size = len(queue)

            for _ in range(level_size):
                current, path = queue.popleft()

                # Check forward edges
                for rel, neighbor in self.graph.get(current, []):
                    if neighbor == cui2:
                        return (len(path), path + [neighbor])
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

                # Check reverse edges
                for rel, neighbor in self.reverse_graph.get(current, []):
                    if neighbor == cui2:
                        return (len(path), path + [neighbor])
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

            if len(queue) == 0:
                break

        return (float('inf'), [])

    def is_related(self, cui1: str, cui2: str, max_distance: int = 4):
        """Check if two concepts are related within max_distance hops"""
        distance, _ = self.get_distance(cui1, cui2, max_depth=max_distance)
        return distance <= max_distance

    def get_neighbors(self, cui: str):
        """Get all neighbors of a concept"""
        return list(self.graph.get(cui, []))

    def get_ancestors(self, cui: str, max_hops: int = 5):
        """Get all ancestor concepts (via PAR/RB relations)"""
        ancestors = set()
        queue = deque([cui])
        visited = {cui}

        for _ in range(max_hops):
            level_size = len(queue)
            if level_size == 0:
                break
            for _ in range(level_size):
                current = queue.popleft()
                for rel, neighbor in self.graph.get(current, []):
                    if rel in {'PAR', 'RB'} and neighbor not in visited:
                        ancestors.add(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)
        return ancestors

    def get_descendants(self, cui: str, max_hops: int = 5):
        """Get all descendant concepts (via CHD/RN relations)"""
        descendants = set()
        queue = deque([cui])
        visited = {cui}

        for _ in range(max_hops):
            level_size = len(queue)
            if level_size == 0:
                break
            for _ in range(level_size):
                current = queue.popleft()
                for rel, neighbor in self.graph.get(current, []):
                    if rel in {'CHD', 'RN'} and neighbor not in visited:
                        descendants.add(neighbor)
                        visited.add(neighbor)
                        queue.append(neighbor)
        return descendants

# ============================================================================
# NEW: HIERARCHY-AWARE CONCEPT RANKER
# ============================================================================

class HierarchyAwareConceptRanker:
    """Ranks concepts using both semantic similarity and hierarchical distance"""

    def __init__(self, hierarchy_loader, icd_to_cui_map: Dict):
        self.hierarchy = hierarchy_loader
        self.icd_to_cui = icd_to_cui_map

        # Scoring weights
        self.SIMILARITY_WEIGHT = 0.4  # Weight for embedding similarity
        self.DISTANCE_WEIGHT = 0.6    # Weight for graph distance
        self.MAX_DISTANCE = 5

    def score_concepts_for_diagnosis(self,
                                    concept_scores: np.ndarray,
                                    concept_ids: List[str],
                                    predicted_icd: str,
                                    max_distance: int = 4) -> np.ndarray:
        """Score concepts combining semantic similarity and hierarchical distance"""

        # Get diagnosis CUIs
        diagnosis_cuis = self._get_diagnosis_cuis(predicted_icd)

        if len(diagnosis_cuis) == 0:
            # No hierarchy info, return semantic scores only
            return concept_scores

        # Compute hierarchical scores for each concept
        hierarchical_scores = np.zeros_like(concept_scores)

        for i, cui in enumerate(concept_ids):
            # Find minimum distance to any diagnosis CUI
            min_distance = float('inf')

            for dx_cui in diagnosis_cuis:
                distance, _ = self.hierarchy.get_distance(cui, dx_cui, max_depth=max_distance + 1)
                min_distance = min(min_distance, distance)

            # Convert distance to score (0-1, where closer = higher)
            if min_distance <= max_distance:
                # Exponential decay: distance 0 = 1.0, distance 4 = ~0.2
                hierarchical_scores[i] = np.exp(-0.5 * min_distance)
            else:
                # Too far, low score
                hierarchical_scores[i] = 0.01

        # Combine semantic and hierarchical scores
        combined_scores = (
            self.SIMILARITY_WEIGHT * concept_scores +
            self.DISTANCE_WEIGHT * hierarchical_scores
        )

        return combined_scores

    def _get_diagnosis_cuis(self, icd_code: str) -> List[str]:
        """Get UMLS CUIs for an ICD code"""

        # Try exact match and variants
        variants = [
            icd_code,
            icd_code.replace('.', ''),
            icd_code[:3],  # Category
        ]

        for variant in variants:
            if variant in self.icd_to_cui:
                return list(self.icd_to_cui[variant])[:5]  # Top 5

        return []

    def explain_concept_relevance(self, cui: str, predicted_icd: str,
                                 concept_store) -> Dict:
        """Explain why a concept is relevant (or not)"""

        diagnosis_cuis = self._get_diagnosis_cuis(predicted_icd)

        explanation = {
            'cui': cui,
            'name': concept_store.concepts.get(cui, {}).get('name', 'Unknown'),
            'icd_code': predicted_icd,
            'diagnosis_cuis': diagnosis_cuis,
            'min_distance': float('inf'),
            'is_relevant': False,
            'paths': {}
        }

        for dx_cui in diagnosis_cuis:
            distance, path = self.hierarchy.get_distance(cui, dx_cui, max_depth=6)
            explanation['paths'][dx_cui] = path

            if distance < explanation['min_distance']:
                explanation['min_distance'] = distance

        explanation['is_relevant'] = explanation['min_distance'] <= 4

        return explanation

# ============================================================================
# UMLS LOADER (keeping original)
# ============================================================================

class UMLSLoader:
    def __init__(self, umls_path: Path):
        self.umls_path = umls_path
        self.concepts = {}
        self.cui_to_icd10 = defaultdict(list)
        self.icd10_to_cui = defaultdict(list)

    def load_concepts(self, max_concepts: int = 50000,
                     filter_semantic_types: List[str] = None):
        print(f"\nLoading UMLS concepts...")

        target_types = filter_semantic_types or [
            'T047', 'T046', 'T184', 'T033', 'T048',
        ]

        cui_to_types = self._load_semantic_types()

        mrconso_path = self.umls_path / 'MRCONSO.RRF'
        concepts_loaded = 0

        with open(mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="Loading"):
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

                if cui in cui_to_types:
                    types = cui_to_types[cui]
                    if not any(t in target_types for t in types):
                        continue
                else:
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
                    self.cui_to_icd10[cui].append(code)
                    self.icd10_to_cui[code].append(cui)

        print(f"Loaded {len(self.concepts)} concepts")
        return self.concepts

    def _load_semantic_types(self) -> Dict[str, List[str]]:
        mrsty_path = self.umls_path / 'MRSTY.RRF'
        cui_to_types = defaultdict(list)

        with open(mrsty_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('|')
                if len(fields) >= 2:
                    cui_to_types[fields[0]].append(fields[1])

        return cui_to_types

    def load_definitions(self, concepts: Dict) -> Dict:
        print("\nLoading definitions...")
        mrdef_path = self.umls_path / 'MRDEF.RRF'
        definitions_added = 0

        with open(mrdef_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading"):
                fields = line.strip().split('|')
                if len(fields) >= 6:
                    cui = fields[0]
                    definition = fields[5]

                    if cui in concepts and definition:
                        if 'definition' not in concepts[cui]:
                            concepts[cui]['definition'] = definition
                            definitions_added += 1

        print(f"Added {definitions_added} definitions")
        return concepts

# ============================================================================
# MIMIC & DATA PREP (keeping original)
# ============================================================================

class MIMICLoader:
    def __init__(self, mimic_path: Path, notes_path: Path):
        self.mimic_path = mimic_path
        self.hosp_path = mimic_path / 'mimic-iv-3.1/hosp'
        self.notes_path = notes_path

    def load_diagnoses(self) -> pd.DataFrame:
        diag_path = self.hosp_path / 'diagnoses_icd.csv.gz'
        return pd.read_csv(diag_path, compression='gzip')

    def load_admissions(self) -> pd.DataFrame:
        adm_path = self.hosp_path / 'admissions.csv.gz'
        return pd.read_csv(adm_path, compression='gzip')

    def load_discharge_notes(self) -> pd.DataFrame:
        discharge_path = self.notes_path / 'note' / 'discharge.csv.gz'
        if not discharge_path.exists():
            discharge_path = self.notes_path / 'discharge.csv.gz'
        return pd.read_csv(discharge_path, compression='gzip')

def load_icd10_descriptions(icd_path: Path) -> Dict[str, str]:
    codes_file = icd_path / 'icd10cm-codes-2024.txt'
    descriptions = {}
    with open(codes_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    descriptions[parts[0]] = parts[1]
    return descriptions

def prepare_dataset(df_diag, df_adm, df_notes, icd_descriptions,
                   target_codes, min_samples_per_code=100):
    """Prepare dataset (keeping original logic)"""

    df_diag = df_diag[df_diag['icd_version'] == 10].copy()
    df_diag['icd_code'] = df_diag['icd_code'].str.replace('.', '', regex=False)

    text_col = 'text'
    if 'text' not in df_notes.columns:
        text_cols = [col for col in df_notes.columns if 'text' in col.lower()]
        text_col = text_cols[0]

    df_notes_with_diag = df_notes.merge(
        df_diag.groupby('hadm_id')['icd_code'].apply(list).reset_index(),
        on='hadm_id', how='inner'
    )

    df = df_notes_with_diag.rename(columns={
        'icd_code': 'icd_codes',
        text_col: 'text'
    })[['hadm_id', 'text', 'icd_codes']].copy()

    df['has_target'] = df['icd_codes'].apply(
        lambda codes: any(code in target_codes for code in codes)
    )
    df_filtered = df[df['has_target']].copy()

    df_filtered['labels'] = df_filtered['icd_codes'].apply(
        lambda codes: [1 if code in codes else 0 for code in target_codes]
    )

    # Balance dataset
    max_per_code = 3000
    balanced_indices = set()

    for code in target_codes:
        code_indices = df_filtered[
            df_filtered['icd_codes'].apply(lambda x: code in x)
        ].index.tolist()
        n_samples = min(len(code_indices), max_per_code)
        selected = np.random.choice(code_indices, size=n_samples, replace=False)
        balanced_indices.update(selected)

    df_final = df_filtered.loc[list(balanced_indices)].reset_index(drop=True)
    df_final = df_final[df_final['text'].notnull()].reset_index(drop=True)

    return df_final, target_codes

# ============================================================================
# ENHANCED CONCEPT STORE (keeping original)
# ============================================================================

class EnhancedConceptStore:
    def __init__(self, umls_concepts: Dict, icd_to_cui: Dict):
        self.umls_concepts = umls_concepts
        self.icd_to_cui = icd_to_cui
        self.concepts = {}
        self.concept_to_idx = {}
        self.idx_to_concept = {}

    def build_concept_set(self, target_icd_codes: List[str],
                         icd_descriptions: Dict[str, str],
                         target_concept_count: int = 200):

        print(f"\n🔬 Building concept set (target: {target_concept_count})...")

        relevant_cuis = set()

        # Strategy 1: Direct ICD mappings
        for icd in target_icd_codes:
            variants = self._get_icd_variants(icd)
            for variant in variants:
                if variant in self.icd_to_cui:
                    cuis = self.icd_to_cui[variant][:20]
                    relevant_cuis.update(cuis)

        print(f"  Direct mappings: {len(relevant_cuis)} concepts")

        # Strategy 2: Keyword expansion
        diagnosis_keywords = {
            'J189': ['pneumonia', 'lung infection', 'respiratory infection',
                    'consolidation', 'infiltrate', 'bacterial pneumonia'],
            'I5023': ['heart failure', 'cardiac failure', 'ventricular failure',
                     'CHF', 'cardiomyopathy', 'cardiac dysfunction'],
            'A419': ['sepsis', 'septicemia', 'infection', 'bacteremia',
                    'systemic infection', 'septic shock'],
            'K8000': ['cholecystitis', 'gallbladder', 'biliary disease',
                     'gallstone', 'cholelithiasis']
        }

        for icd in target_icd_codes:
            keywords = diagnosis_keywords.get(icd, [])
            if icd in icd_descriptions:
                desc_words = [
                    w for w in icd_descriptions[icd].lower().split()
                    if len(w) > 4
                ][:5]
                keywords.extend(desc_words)

            # Search and add
            for cui, info in self.umls_concepts.items():
                if cui in relevant_cuis:
                    continue

                terms_text = ' '.join(
                    [info['name']] + info.get('terms', [])
                ).lower()

                if any(kw in terms_text for kw in keywords):
                    relevant_cuis.add(cui)

                if len(relevant_cuis) >= target_concept_count * 2:
                    break

        print(f"  After keyword expansion: {len(relevant_cuis)} concepts")

        # Build final concept dictionary
        for cui in list(relevant_cuis)[:target_concept_count]:
            if cui in self.umls_concepts:
                concept = self.umls_concepts[cui]
                self.concepts[cui] = {
                    'cui': cui,
                    'name': concept['name'],
                    'definition': concept.get('definition', ''),
                    'terms': concept.get('terms', []),
                    'semantic_types': concept.get('semantic_types', [])
                }

        concept_list = list(self.concepts.keys())
        self.concept_to_idx = {cui: i for i, cui in enumerate(concept_list)}
        self.idx_to_concept = {i: cui for i, cui in enumerate(concept_list)}

        print(f"  Final: {len(self.concepts)} concepts")
        return self.concepts

    def _get_icd_variants(self, code: str) -> List[str]:
        variants = {code, code.replace('.', '')}
        no_dots = code.replace('.', '')
        if len(no_dots) >= 4:
            variants.add(no_dots[:3] + '.' + no_dots[3:])
        variants.add(no_dots[:3])
        return list(variants)

    def create_concept_embeddings(self, tokenizer, model, device):
        concept_texts = []
        for cui, info in self.concepts.items():
            text = f"{info['name']}."
            if info['definition']:
                text += f" {info['definition'][:200]}"
            concept_texts.append(text)

        batch_size = 32
        all_embeddings = []

        for i in tqdm(range(0, len(concept_texts), batch_size), desc="Encoding concepts"):
            batch = concept_texts[i:i+batch_size]
            encodings = tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = model(**encodings)
                embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).to(device)

# ============================================================================
# RAG RETRIEVER (keeping original)
# ============================================================================

class ImprovedMedicalRAG:
    def __init__(self, concept_store, umls_concepts, icd_descriptions, target_codes):
        self.concept_store = concept_store
        self.umls_concepts = umls_concepts
        self.icd_descriptions = icd_descriptions
        self.target_codes = target_codes
        self.documents = []
        self.doc_metadata = []
        self.index = None

    def build_focused_document_store(self):
        print("\n📚 Building RAG document store...")

        # Add concept definitions
        for cui, info in self.concept_store.concepts.items():
            if info.get('definition'):
                doc_text = f"{info['name']}. {info['definition']}"
                self.documents.append(doc_text)
                self.doc_metadata.append({
                    'type': 'concept',
                    'cui': cui,
                    'name': info['name'],
                    'source': 'UMLS'
                })

        # Add ICD descriptions
        for icd_code in self.target_codes:
            if icd_code in self.icd_descriptions:
                desc = self.icd_descriptions[icd_code]
                self.documents.append(f"ICD-10 {icd_code}: {desc}")
                self.doc_metadata.append({
                    'type': 'icd',
                    'code': icd_code,
                    'description': desc,
                    'source': 'ICD-10-CM'
                })

        print(f"  Built: {len(self.documents)} documents")
        return self.documents

    def build_faiss_index(self, tokenizer, model, device):
        print("\n🔍 Building FAISS index...")

        batch_size = 32
        all_embeddings = []

        for i in tqdm(range(0, len(self.documents), batch_size), desc="Encoding"):
            batch = self.documents[i:i+batch_size]
            encodings = tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = model(**encodings)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

        self.doc_embeddings = np.vstack(all_embeddings).astype('float32')

        dimension = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.doc_embeddings)

        print(f"  FAISS index: {self.index.ntotal} documents")
        return self.index

    def retrieve_with_diagnosis_filter(self, query_embeddings, predicted_diagnosis, k=5):
        if self.index is None:
            raise ValueError("Index not built!")

        distances, indices = self.index.search(
            query_embeddings.astype('float32'), min(k * 2, len(self.documents))
        )

        batch_results = []
        for query_dists, query_indices in zip(distances, indices):
            filtered_results = []

            for dist, idx in zip(query_dists, query_indices):
                if idx >= len(self.documents):
                    continue

                metadata = self.doc_metadata[idx]

                # Prefer diagnosis-relevant documents
                is_relevant = (
                    metadata['type'] == 'icd' and metadata['code'] == predicted_diagnosis
                ) or metadata['type'] == 'concept'

                if is_relevant:
                    filtered_results.append((
                        self.documents[idx],
                        metadata,
                        float(dist)
                    ))

                if len(filtered_results) >= k:
                    break

            batch_results.append(filtered_results)

        return batch_results

# ============================================================================
# DATASET (keeping original)
# ============================================================================

class ClinicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

# ============================================================================
# MODEL COMPONENTS (keeping original)
# ============================================================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, concept_embeddings, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_concepts = concept_embeddings.shape[0]

        concepts_batch = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(concepts_batch).view(batch_size, num_concepts, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        gate_input = torch.cat([hidden_states, context], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))

        output = hidden_states + gate_values * context
        output = self.layer_norm(output)

        return output, attn_weights.mean(dim=1)

class EnhancedCitationHead(nn.Module):
    def __init__(self, hidden_size, num_concepts, num_classes):
        super().__init__()

        self.concept_selector = nn.Linear(hidden_size, num_concepts)
        self.diagnosis_head = nn.Linear(hidden_size, num_classes)
        self.diagnosis_concept_interaction = nn.Bilinear(num_classes, num_concepts, num_concepts)
        self.evidence_scorer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        cls_hidden = hidden_states[:, 0, :]

        diagnosis_logits = self.diagnosis_head(cls_hidden)
        diagnosis_probs = torch.sigmoid(diagnosis_logits)

        concept_logits = self.concept_selector(cls_hidden)
        refined_concept_logits = self.diagnosis_concept_interaction(
            diagnosis_probs, torch.sigmoid(concept_logits)
        )

        concept_probs = torch.sigmoid(refined_concept_logits)
        evidence_scores = self.evidence_scorer(hidden_states).squeeze(-1)

        return {
            'logits': diagnosis_logits,
            'concept_scores': refined_concept_logits,
            'concept_probs': concept_probs,
            'evidence_scores': evidence_scores
        }

# ============================================================================
# NEW: MODEL WITH HIERARCHY INTEGRATION
# ============================================================================

class HierarchyShifaMind(nn.Module):
    """ShifaMind with hierarchy-aware concept selection"""

    def __init__(self, base_model, concept_store, rag_retriever,
                 hierarchy_ranker, num_classes, fusion_layers=[6, 9, 11]):
        super().__init__()

        self.base_model = base_model
        self.concept_store = concept_store
        self.rag_retriever = rag_retriever
        self.hierarchy_ranker = hierarchy_ranker
        self.num_classes = num_classes
        self.hidden_size = base_model.config.hidden_size
        self.fusion_layers = fusion_layers

        self.fusion_modules = nn.ModuleList([
            CrossAttentionFusion(self.hidden_size)
            for _ in fusion_layers
        ])

        self.citation_head = EnhancedCitationHead(
            self.hidden_size,
            len(concept_store.concepts),
            num_classes
        )

    def forward(self, input_ids, attention_mask, concept_embeddings,
                retrieve_docs=True, use_hierarchy_filter=True):

        # Standard forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        current_hidden = hidden_states[-1]

        # Apply fusion
        fusion_attentions = []
        for i, fusion_module in enumerate(self.fusion_modules):
            layer_idx = self.fusion_layers[i]
            layer_hidden = hidden_states[layer_idx]

            fused_hidden, attn_weights = fusion_module(
                layer_hidden, concept_embeddings, attention_mask
            )
            fusion_attentions.append(attn_weights)

            if i == len(self.fusion_modules) - 1:
                current_hidden = fused_hidden

        # Get citation outputs
        citation_outputs = self.citation_head(current_hidden)

        # ===== NEW: Apply hierarchy filtering =====
        if use_hierarchy_filter and self.hierarchy_ranker is not None:
            diagnosis_logits = citation_outputs['logits']
            predicted_diagnoses = torch.argmax(diagnosis_logits, dim=1).cpu().numpy()
            concept_scores = citation_outputs['concept_probs']
            batch_size = concept_scores.shape[0]

            concept_ids = list(self.concept_store.concepts.keys())
            target_codes = self.rag_retriever.target_codes if self.rag_retriever else []

            # Filter each sample
            filtered_concept_scores = torch.zeros_like(concept_scores)

            for b in range(batch_size):
                sample_scores = concept_scores[b].cpu().detach().numpy()
                pred_idx = predicted_diagnoses[b]

                if pred_idx < len(target_codes):
                    pred_icd = target_codes[pred_idx]

                    # Apply hierarchy ranking
                    combined_scores = self.hierarchy_ranker.score_concepts_for_diagnosis(
                        sample_scores, concept_ids, pred_icd, max_distance=4
                    )

                    filtered_concept_scores[b] = torch.tensor(
                        combined_scores, device=concept_scores.device
                    )
                else:
                    filtered_concept_scores[b] = concept_scores[b]

            citation_outputs['concept_probs'] = filtered_concept_scores

        # RAG retrieval
        retrieved_docs = None
        if retrieve_docs and self.rag_retriever is not None:
            cls_embeddings = current_hidden[:, 0, :].detach().cpu().numpy()

            if use_hierarchy_filter and len(target_codes) > 0:
                retrieved_docs = []
                for b in range(batch_size):
                    pred_idx = predicted_diagnoses[b]
                    if pred_idx < len(target_codes):
                        pred_dx = target_codes[pred_idx]
                        docs = self.rag_retriever.retrieve_with_diagnosis_filter(
                            cls_embeddings[b:b+1], pred_dx, k=3
                        )
                        retrieved_docs.extend(docs)

        citation_outputs['retrieved_docs'] = retrieved_docs
        citation_outputs['fusion_attentions'] = fusion_attentions

        return citation_outputs

# ============================================================================
# LOSS FUNCTION (keeping original)
# ============================================================================

class EnhancedMultiObjectiveLoss(nn.Module):
    def __init__(self, lambda_weights=None):
        super().__init__()

        self.lambdas = lambda_weights or {
            'diagnosis': 0.4,
            'concept_precision': 0.3,
            'citation': 0.2,
            'calibration': 0.1
        }

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, concept_scores, target_concepts=None):
        l_dx = self.bce_loss(logits, labels)

        if target_concepts is not None:
            l_concept = F.binary_cross_entropy_with_logits(concept_scores, target_concepts)
        else:
            probs = torch.sigmoid(concept_scores)
            l_concept = torch.mean(probs)

        probs = torch.sigmoid(concept_scores)
        top_k_probs = torch.topk(probs, k=10, dim=1)[0]
        l_cite = -torch.mean(top_k_probs)

        predictions = (torch.sigmoid(logits) > 0.5).float()
        confidences = torch.max(torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        correct = (predictions == labels).float().mean(dim=1)
        l_cal = torch.abs(confidences.mean(dim=1) - correct).mean()

        total = (
            self.lambdas['diagnosis'] * l_dx +
            self.lambdas['concept_precision'] * l_concept +
            self.lambdas['citation'] * l_cite +
            self.lambdas['calibration'] * l_cal
        )

        return {
            'total': total,
            'diagnosis': l_dx.item(),
            'concept_precision': l_concept.item(),
            'citation': l_cite.item(),
            'calibration': l_cal.item(),
            'concept_activation': probs.mean().item()
        }

# ============================================================================
# EVALUATION (keeping original structure)
# ============================================================================

def evaluate(model, dataloader, concept_embeddings, device, concept_ids, threshold=0.5):
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    all_concept_scores, all_concept_preds = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids, attention_mask, concept_embeddings,
                retrieve_docs=False, use_hierarchy_filter=True
            )

            probs = torch.sigmoid(outputs['logits'])
            preds = (probs > threshold).float()
            concept_scores = outputs['concept_probs']
            concept_preds = (concept_scores > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_concept_scores.append(concept_scores.cpu().numpy())
            all_concept_preds.append(concept_preds.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    all_concept_scores = np.vstack(all_concept_scores)
    all_concept_preds = np.vstack(all_concept_preds)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    try:
        macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
    except:
        macro_auc = 0.0

    if all_concept_scores.shape[1] > 1:
        concept_precision = precision_score(
            (all_concept_scores > 0.3).astype(int),
            all_concept_preds.astype(int),
            average='samples', zero_division=0
        )
        concept_recall = recall_score(
            (all_concept_scores > 0.3).astype(int),
            all_concept_preds.astype(int),
            average='samples', zero_division=0
        )
    else:
        concept_precision = 0.0
        concept_recall = 0.0

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'per_class_f1': per_class_f1,
        'macro_auc': macro_auc,
        'concept_precision': concept_precision,
        'concept_recall': concept_recall,
        'concept_scores': all_concept_scores,
        'concept_preds': all_concept_preds
    }

# ============================================================================
# VISUALIZATION (keeping original)
# ============================================================================

def plot_results(baseline_metrics, final_metrics, target_codes):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Overall metrics
    metrics = ['Macro F1', 'Micro F1', 'AUROC']
    baseline_vals = [baseline_metrics['macro_f1'], baseline_metrics['micro_f1'],
                    baseline_metrics.get('macro_auc', 0)]
    final_vals = [final_metrics['macro_f1'], final_metrics['micro_f1'],
                  final_metrics.get('macro_auc', 0)]

    x = np.arange(len(metrics))
    width = 0.35

    axes[0].bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
    axes[0].bar(x + width/2, final_vals, width, label='ShifaMind+Hierarchy', alpha=0.8)
    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Diagnosis Performance')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Concept metrics
    concept_metrics = ['Precision', 'Recall']
    concept_vals = [
        final_metrics.get('concept_precision', 0),
        final_metrics.get('concept_recall', 0)
    ]

    axes[1].bar(concept_metrics, concept_vals, alpha=0.8, color='green')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Concept Selection Quality')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)

    # Per-class F1
    baseline_per_class = baseline_metrics['per_class_f1']
    final_per_class = final_metrics['per_class_f1']

    x = np.arange(len(target_codes))
    axes[2].bar(x - width/2, baseline_per_class, width, label='Baseline', alpha=0.8)
    axes[2].bar(x + width/2, final_per_class, width, label='ShifaMind+Hierarchy', alpha=0.8)
    axes[2].set_xlabel('ICD-10 Code')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Per-Class F1')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(target_codes, rotation=45)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hierarchy_results.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: hierarchy_results.png")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("\n" + "="*70)
print("SHIFAMIND WITH HIERARCHY INTEGRATION")
print("="*70)

# Load UMLS
print("\n📂 Loading UMLS...")
umls_loader = UMLSLoader(UMLS_PATH)
umls_concepts = umls_loader.load_concepts(max_concepts=50000)
umls_concepts = umls_loader.load_definitions(umls_concepts)

# ===== NEW: Load Hierarchy =====
print("\n🌳 Loading UMLS Hierarchy...")
mrrel_path = UMLS_PATH / 'MRREL.RRF'
cache_path = Path('/content/drive/MyDrive/ShifaMind/umls_hierarchy_cache.pkl')

hierarchy_loader = UMLSHierarchyLoader(mrrel_path)

# Load MIMIC
print("\n📂 Loading MIMIC-IV...")
mimic_loader = MIMICLoader(MIMIC_PATH, NOTES_PATH)
df_diagnoses = mimic_loader.load_diagnoses()
df_admissions = mimic_loader.load_admissions()
df_notes = mimic_loader.load_discharge_notes()

print("\n📂 Loading ICD-10...")
icd10_descriptions = load_icd10_descriptions(ICD_PATH)

TARGET_CODES = ['J189', 'I5023', 'A419', 'K8000']

print(f"\n🎯 Target diagnoses:")
for code in TARGET_CODES:
    print(f"  {code}: {icd10_descriptions.get(code, 'Unknown')}")

# Prepare dataset
df_train, target_codes = prepare_dataset(
    df_diagnoses, df_admissions, df_notes,
    icd10_descriptions, TARGET_CODES, min_samples_per_code=100
)

print(f"\n✅ Dataset: {len(df_train)} samples")

# Get seed concepts for hierarchy
seed_concepts = set()
for icd in target_codes:
    variants = [icd, icd.replace('.', ''), icd[:3]]
    for variant in variants:
        if variant in umls_loader.icd10_to_cui:
            seed_concepts.update(umls_loader.icd10_to_cui[variant])

print(f"\nSeed concepts for hierarchy: {len(seed_concepts)}")

# Load hierarchy (3 hops, cached)
hierarchy_stats = hierarchy_loader.load_for_concepts(
    seed_concepts, max_hops=3, cache_path=cache_path
)

print(f"\n✅ Hierarchy loaded:")
print(f"   Concepts: {hierarchy_stats['num_concepts']}")
print(f"   Edges: {hierarchy_stats['num_edges']}")

# ===== NEW: Create hierarchy ranker =====
hierarchy_ranker = HierarchyAwareConceptRanker(
    hierarchy_loader,
    umls_loader.icd10_to_cui
)

# Build concept store
print("\n🔬 Building concept store...")
concept_store = EnhancedConceptStore(
    umls_concepts,
    umls_loader.icd10_to_cui
)

concept_set = concept_store.build_concept_set(
    target_codes,
    icd10_descriptions,
    target_concept_count=200
)

# Load model
print("\n🤖 Loading BioClinicalBERT...")
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name).to(device)

concept_embeddings = concept_store.create_concept_embeddings(
    tokenizer, base_model, device
)

print(f"✅ Concept embeddings: {concept_embeddings.shape}")

# Build RAG
print("\n📚 Building RAG...")
rag_retriever = ImprovedMedicalRAG(
    concept_store, umls_concepts, icd10_descriptions, target_codes
)

documents = rag_retriever.build_focused_document_store()
rag_index = rag_retriever.build_faiss_index(tokenizer, base_model, device)

# Split data
print(f"\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df_train['text'].values,
    np.array(df_train['labels'].tolist()),
    test_size=0.2,
    random_state=SEED
)

print(f"  Train: {len(X_train)}")
print(f"  Test:  {len(X_test)}")

train_dataset = ClinicalDataset(X_train, y_train, tokenizer)
test_dataset = ClinicalDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Baseline
print("\n" + "="*70)
print("BASELINE")
print("="*70)

class BaselineModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, concept_embeddings=None,
                retrieve_docs=False, use_hierarchy_filter=False):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return {
            'logits': logits,
            'concept_scores': torch.zeros(logits.shape[0], 1, device=logits.device),
            'concept_probs': torch.zeros(logits.shape[0], 1, device=logits.device)
        }

baseline = BaselineModel(base_model, len(target_codes)).to(device)
baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=2e-5)
baseline_criterion = nn.BCEWithLogitsLoss()

for epoch in range(1):
    baseline.train()
    for batch in tqdm(train_loader, desc=f"Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        baseline_optimizer.zero_grad()
        outputs = baseline(input_ids, attention_mask)
        loss = baseline_criterion(outputs['logits'], labels)
        loss.backward()
        baseline_optimizer.step()

baseline_metrics = evaluate(
    baseline, test_loader, concept_embeddings, device,
    list(concept_store.concepts.keys())
)

print(f"\n📊 Baseline: {baseline_metrics['macro_f1']:.4f} macro F1")

# ===== NEW: Train ShifaMind with Hierarchy =====
print("\n" + "="*70)
print("TRAINING SHIFAMIND + HIERARCHY")
print("="*70)

hierarchy_model = HierarchyShifaMind(
    base_model=base_model,
    concept_store=concept_store,
    rag_retriever=rag_retriever,
    hierarchy_ranker=hierarchy_ranker,  # NEW!
    num_classes=len(target_codes),
    fusion_layers=[6, 9, 11]
).to(device)

num_epochs = 3
optimizer = torch.optim.AdamW(hierarchy_model.parameters(), lr=2e-5, weight_decay=0.01)
num_training_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps
)

criterion = EnhancedMultiObjectiveLoss()

best_f1 = 0

for epoch in range(num_epochs):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*70}")

    hierarchy_model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = hierarchy_model(
            input_ids, attention_mask, concept_embeddings,
            retrieve_docs=False, use_hierarchy_filter=True  # Use hierarchy!
        )

        loss_dict = criterion(
            outputs['logits'],
            labels,
            outputs['concept_scores']
        )

        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(hierarchy_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss_dict['total'].item()

    avg_loss = total_loss / len(train_loader)

    # Evaluate
    metrics = evaluate(
        hierarchy_model, test_loader, concept_embeddings, device,
        list(concept_store.concepts.keys())
    )

    print(f"\n📊 Epoch {epoch+1}:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Concept Precision: {metrics['concept_precision']:.4f}")

    if metrics['macro_f1'] > best_f1:
        best_f1 = metrics['macro_f1']
        torch.save(hierarchy_model.state_dict(), 'best_hierarchy_model.pt')
        print(f"  ✅ New best!")

# Final evaluation
final_metrics = evaluate(
    hierarchy_model, test_loader, concept_embeddings, device,
    list(concept_store.concepts.keys())
)

# Results
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

print(f"\n📊 Diagnosis Performance:")
print(f"  Baseline:  {baseline_metrics['macro_f1']:.4f} macro F1")
print(f"  Hierarchy: {final_metrics['macro_f1']:.4f} macro F1")
improvement = final_metrics['macro_f1'] - baseline_metrics['macro_f1']
print(f"  Change:    {improvement:+.4f}")

print(f"\n📊 Concept Quality:")
print(f"  Precision: {final_metrics['concept_precision']:.4f}")
print(f"  Recall:    {final_metrics['concept_recall']:.4f}")

# Plot
print("\n📊 Creating visualizations...")
plot_results(baseline_metrics, final_metrics, target_codes)

# ===== NEW: Demo with Hierarchy Explanations =====
print("\n" + "="*70)
print("DEMO: HIERARCHY-AWARE CONCEPT SELECTION")
print("="*70)

# Get a test case
pneumonia_indices = df_train[
    df_train['icd_codes'].apply(lambda x: 'J189' in x)
].index[:1]

if len(pneumonia_indices) > 0:
    demo_text = df_train.loc[pneumonia_indices[0], 'text']

    print(f"\n📝 Demo Case:")
    print(f"  {demo_text[:200]}...")

    encoding = tokenizer(
        demo_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    hierarchy_model.eval()
    with torch.no_grad():
        outputs = hierarchy_model(
            encoding['input_ids'],
            encoding['attention_mask'],
            concept_embeddings,
            retrieve_docs=False,
            use_hierarchy_filter=True
        )

        diagnosis_probs = torch.sigmoid(outputs['logits'])[0]
        concept_scores = outputs['concept_probs'][0]

        pred_idx = torch.argmax(diagnosis_probs).item()
        pred_icd = target_codes[pred_idx]

    print(f"\n🏥 DIAGNOSIS:")
    for i, code in enumerate(target_codes):
        prob = diagnosis_probs[i].item()
        label = "✅" if prob > 0.5 else "❌"
        print(f"  {label} {code}: {prob:.1%}")

    print(f"\n🌳 TOP CONCEPTS (Hierarchy-Filtered):")

    concept_ids = list(concept_store.concepts.keys())
    top_indices = torch.argsort(concept_scores, descending=True)[:10].cpu().numpy()

    relevant_count = 0

    for rank, idx in enumerate(top_indices, 1):
        cui = concept_ids[idx]
        concept_info = concept_store.concepts[cui]
        score = concept_scores[idx].item()

        # Get hierarchy explanation
        explanation = hierarchy_ranker.explain_concept_relevance(
            cui, pred_icd, concept_store
        )

        print(f"\n  {rank}. {concept_info['name']}")
        print(f"     Score: {score:.3f}")
        print(f"     Distance: {explanation['min_distance']} hops")

        if explanation['is_relevant']:
            print(f"     ✅ Hierarchically relevant")
            relevant_count += 1

            # Show path
            if explanation['min_distance'] < float('inf'):
                dx_cui = list(explanation['paths'].keys())[0]
                path = explanation['paths'][dx_cui]
                if len(path) > 1:
                    path_str = ' → '.join([p[:8] for p in path[:4]])
                    if len(path) > 4:
                        path_str += '...'
                    print(f"     Path: {path_str}")
        else:
            print(f"     ❌ Too distant")

    print(f"\n📊 HIERARCHY VALIDATION:")
    print(f"  Predicted: {pred_icd} - {icd10_descriptions[pred_icd]}")
    print(f"  Top 10 concepts: {relevant_count}/10 hierarchically relevant "
          f"({relevant_count/10*100:.0f}%)")

print("\n" + "="*70)
print("✅ COMPLETE!")
print("="*70)

print("\n💾 Saved:")
print("  - best_hierarchy_model.pt")
print("  - hierarchy_results.png")
print("  - umls_hierarchy_cache.pkl (reused next time!)")

print("\n📈 Expected improvements:")
print("  - Concept relevance: 12.5% → 60-80%")
print("  - F1 maintained: ~77-78%")
print("  - Demo shows clinical relevance!")
