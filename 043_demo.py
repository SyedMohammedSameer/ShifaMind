#!/usr/bin/env python3
"""
ShifaMind 043: Interactive Demo

Note: This is a template. To run, you need to:
1. Install gradio: pip install gradio
2. Run 043.py to create checkpoint
3. Run generate_knowledge_base_043.py to create KB

Author: Mohammed Sameer Syed
"""

import gradio as gr
import json
from pathlib import Path
from datetime import datetime

print("ShifaMind 043 Demo - Template")
print("This demonstrates the Phase 2 interface structure")

def predict_demo(clinical_note):
    """Demo prediction function"""
    
    if not clinical_note or len(clinical_note) < 10:
        return "⚠️ Please enter a clinical note", "", "", "", "{}"
    
    # Demo output
    diagnosis = "## 🎯 Pneumonia, unspecified organism\n**Code:** J189  \n**Confidence:** 87.3%"
    
    confidence = "### All Probabilities:\n\n**J189**: 87.3% ████████████████\n**I5023**: 5.2% █\n**A419**: 4.1% \n**K8000**: 3.4% "
    
    evidence = "## 📋 Evidence Chains\n\n### 1. Pneumonia (89.1%)\n> \"fever of 38.9°C for 3 days\"\n> \"productive cough with yellow sputum\"\n> \"crackles in right lower lobe\"\n\n### 2. Fever (83.4%)\n> \"fever of 38.9°C\"\n> \"elevated temperature\""
    
    knowledge = "## 📚 Clinical Knowledge\n\n### 1. Diagnosis Description\nPneumonia, unspecified organism (J189) is an acute infection of the lung parenchyma.\n\n*Source: ICD-10-CM J189*\n\n### 2. Clinical Presentation\nCommon symptoms include fever, cough, dyspnea, sputum production, and chest pain.\n\n*Source: UMLS C0032285*"
    
    json_out = {
        'diagnosis': {'code': 'J189', 'name': 'Pneumonia', 'confidence': 0.873},
        'reasoning_chain': [
            {'concept': 'Pneumonia', 'score': 0.891, 'evidence_spans': ['fever of 38.9°C']}
        ],
        'clinical_knowledge': [
            {'text': 'Pneumonia is an acute infection...', 'source': 'ICD-10'}
        ],
        'metadata': {'model_version': 'ShifaMind-043', 'timestamp': datetime.now().isoformat()}
    }
    
    return diagnosis, confidence, evidence, knowledge, json.dumps(json_out, indent=2)

# Gradio interface
demo = gr.Interface(
    fn=predict_demo,
    inputs=gr.Textbox(label="Clinical Note", lines=10, placeholder="Enter clinical note..."),
    outputs=[
        gr.Markdown(label="Diagnosis"),
        gr.Markdown(label="Confidence"),
        gr.Markdown(label="Evidence Chains"),
        gr.Markdown(label="Clinical Knowledge"),
        gr.Code(label="JSON Output", language="json")
    ],
    title="🏥 ShifaMind 043: Evidence & Knowledge Demo",
    description="Phase 2 demonstration of evidence extraction and clinical knowledge retrieval",
    examples=[
        ["72-year-old male with fever, cough, dyspnea, and crackles in right lower lobe. CXR shows infiltrate."]
    ]
)

if __name__ == '__main__':
    print("\nLaunching demo interface...")
    print("Note: This is a demo template showing Phase 2 UI structure")
    demo.launch(share=True)
