#!/usr/bin/env python3
"""
ShifaMind 043: Phase 2 Evaluation

Simplified evaluation demonstrating Phase 2 capabilities.
Author: Mohammed Sameer Syed
"""

import torch
import json
from pathlib import Path
from datetime import datetime

BASE_PATH = Path('/content/drive/MyDrive/ShifaMind')
OUTPUT_PATH = BASE_PATH / '04_Results/experiments/043_evidence_rag'
CHECKPOINT_043 = BASE_PATH / '03_Models/checkpoints/shifamind_043_final.pt'
KB_PATH = BASE_PATH / '03_Models/clinical_knowledge_base_043.json'

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("="*80)
print("SHIFAMIND 043: PHASE 2 EVALUATION")
print("="*80)

def main():
    print("\n📂 Loading checkpoint...")
    
    if not CHECKPOINT_043.exists():
        print(f"❌ Checkpoint not found: {CHECKPOINT_043}")
        print("   Run 043.py first")
        return
    
    checkpoint = torch.load(CHECKPOINT_043, map_location='cpu')
    
    # Extract test results
    test_results = checkpoint.get('test_results', {})
    
    print(f"\n📊 Phase 2 Evaluation Results:")
    print(f"  Model Version: ShifaMind-043")
    print(f"  Phase: Evidence Extraction + Knowledge Retrieval")
    
    if test_results:
        print(f"\n  Test Case: {test_results.get('test_case', 'N/A')}")
        print(f"  Predicted Diagnosis: {test_results.get('predicted_diagnosis', 'N/A')}")
        print(f"  Confidence: {test_results.get('confidence', 0):.1%}")
        print(f"  Evidence Chains: {test_results.get('num_evidence_chains', 0)}")
        
        sample = test_results.get('sample_evidence')
        if sample:
            print(f"\n  Sample Evidence:")
            print(f"    Concept: {sample.get('concept', 'N/A')}")
            print(f"    Score: {sample.get('score', 0):.1%}")
            print(f"    Spans: {len(sample.get('evidence_spans', []))}")
    
    # Save metrics
    metrics = {
        'model_version': 'ShifaMind-043',
        'evaluation_date': datetime.now().isoformat(),
        'phase': 'Phase 2',
        'test_results': test_results
    }
    
    with open(OUTPUT_PATH / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Saved results to: {OUTPUT_PATH}")
    print("\n✅ Phase 2 evaluation complete!")

if __name__ == '__main__':
    main()
