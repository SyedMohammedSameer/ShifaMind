# ShifaMind Usage Guide

This guide provides comprehensive examples for using ShifaMind in various scenarios.

## Table of Contents

- [Quick Start](#quick-start)
- [Inference](#inference)
- [Training](#training)
- [Evaluation](#evaluation)
- [Demo Interface](#demo-interface)
- [Python API](#python-api)
- [Command-Line Usage](#command-line-usage)
- [Integration Examples](#integration-examples)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Example

```python
from final_inference import ShifaMindPredictor

# Initialize predictor
predictor = ShifaMindPredictor()

# Predict
result = predictor.predict("""
72-year-old male presents with fever (38.9°C), productive cough,
and shortness of breath. Chest X-ray shows right lower lobe infiltrate.
""")

# Display
print(f"Diagnosis: {result['diagnosis']['name']}")
print(f"Confidence: {result['diagnosis']['confidence']:.1%}")
```

---

## Inference

### Single Prediction

```python
from final_inference import ShifaMindPredictor

# Initialize
predictor = ShifaMindPredictor(
    checkpoint_path='/path/to/checkpoint.pt',  # Optional
    device='cuda'  # Or 'cpu'
)

# Predict
clinical_note = """
Patient presents with severe right upper quadrant pain,
fever, and positive Murphy's sign. Ultrasound shows
gallbladder wall thickening and multiple stones.
"""

result = predictor.predict(clinical_note)

# Access results
diagnosis = result['diagnosis']
print(f"Code: {diagnosis['code']}")
print(f"Name: {diagnosis['name']}")
print(f"Confidence: {diagnosis['confidence']:.2%}")

# Access concepts
for concept in result['concepts'][:5]:
    print(f"- {concept['name']} ({concept['score']:.1%})")
```

### Batch Prediction

```python
# List of clinical notes
notes = [
    "Patient 1 with fever and cough...",
    "Patient 2 with dyspnea and edema...",
    "Patient 3 with abdominal pain...",
]

# Predict on all
results = predictor.predict_batch(notes, batch_size=8)

# Process results
for i, result in enumerate(results):
    print(f"Patient {i+1}: {result['diagnosis']['code']}")
```

### Without Concepts (Faster)

```python
# Disable concept prediction for faster inference
result = predictor.predict(
    clinical_note,
    return_concepts=False  # Faster, no explainability
)
```

---

## Training

### Full Training Pipeline

```python
# Step 1: Generate knowledge base
import subprocess
subprocess.run(['python', 'final_knowledge_base_generator.py'])

# Step 2: Train model
subprocess.run(['python', 'final_model_training.py'])
```

### Training from Python

```python
# Import training module
import sys
sys.path.append('/path/to/ShifaMind')

from final_model_training import train_model

# Train (this will take several hours)
train_model(
    output_path='/path/to/output',
    num_epochs_stage1=3,
    num_epochs_stage2=2,
    num_epochs_stage3=3,
    batch_size=8,
    learning_rate=2e-5
)
```

### Resume Training

```python
from final_model_training import train_model

# Resume from checkpoint
train_model(
    resume_from_checkpoint='/path/to/checkpoint.pt',
    output_path='/path/to/output'
)
```

---

## Evaluation

### Run Evaluation

```bash
python final_evaluation.py
```

### Custom Evaluation

```python
from final_evaluation import evaluate_model

results = evaluate_model(
    checkpoint_path='/path/to/checkpoint.pt',
    test_data_path='/path/to/test_data.pkl',
    output_dir='/path/to/results'
)

# Access metrics
print(f"Macro F1: {results['macro_f1']:.4f}")
print(f"AUROC: {results['macro_auroc']:.4f}")
print(f"Citation Completeness: {results['citation_completeness']:.2%}")
```

---

## Demo Interface

### Launch Demo

```bash
python final_demo.py
```

The demo will launch at `http://localhost:7860`

### Custom Demo Configuration

```python
# Launch with custom settings
import gradio as gr
from final_demo import create_demo_interface

demo = create_demo_interface(
    checkpoint_path='/path/to/checkpoint.pt',
    knowledge_base_path='/path/to/kb.json'
)

demo.launch(
    server_port=8080,
    share=True,  # Create public link
    auth=("username", "password")  # Optional authentication
)
```

---

## Python API

### Complete API Example

```python
from final_inference import ShifaMindPredictor
import json

# Initialize
predictor = ShifaMindPredictor()

# Predict with full output
result = predictor.predict(
    text="Patient with fever, cough, and infiltrate...",
    return_concepts=True,
    top_k_concepts=10
)

# Save to JSON
with open('prediction.json', 'w') as f:
    json.dump(result, f, indent=2)

# Access all fields
print(json.dumps(result, indent=2))
```

### Result Structure

```python
{
    "diagnosis": {
        "code": "J189",
        "name": "Pneumonia, unspecified organism",
        "confidence": 0.94
    },
    "all_probabilities": {
        "J189": 0.94,
        "I5023": 0.03,
        "A419": 0.02,
        "K8000": 0.01
    },
    "concepts": [
        {
            "idx": 42,
            "cui": "C0032285",
            "name": "Pneumonia",
            "score": 0.92,
            "semantic_types": ["Disease or Syndrome"]
        },
        # ... more concepts
    ],
    "metadata": {
        "model_version": "ShifaMind",
        "timestamp": "2025-11-20T10:30:00",
        "text_length": 150,
        "device": "cuda",
        "num_concepts": 10
    }
}
```

---

## Command-Line Usage

### Basic Prediction

```bash
python final_inference.py --text "Patient with fever and cough..." --format text
```

### Predict from File

```bash
python final_inference.py --file clinical_note.txt --format detailed
```

### Save Output

```bash
python final_inference.py --file note.txt --format json --save result.json
```

### Disable Concepts (Faster)

```bash
python final_inference.py --text "..." --no-concepts --format text
```

### Custom Checkpoint

```bash
python final_inference.py \
    --text "..." \
    --checkpoint /path/to/custom_checkpoint.pt \
    --format detailed
```

---

## Integration Examples

### REST API Integration (Flask)

```python
from flask import Flask, request, jsonify
from final_inference import ShifaMindPredictor

app = Flask(__name__)
predictor = ShifaMindPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    clinical_note = data.get('text')

    if not clinical_note:
        return jsonify({'error': 'No text provided'}), 400

    result = predictor.predict(clinical_note)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from final_inference import ShifaMindPredictor

app = FastAPI()
predictor = ShifaMindPredictor()

class PredictionRequest(BaseModel):
    text: str
    return_concepts: bool = True

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        result = predictor.predict(
            request.text,
            return_concepts=request.return_concepts
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Jupyter Notebook Integration

```python
# In Jupyter Notebook
from final_inference import ShifaMindPredictor
import pandas as pd

# Initialize
predictor = ShifaMindPredictor()

# Load test data
df = pd.read_csv('test_notes.csv')

# Predict on all
results = []
for text in df['clinical_note']:
    result = predictor.predict(text)
    results.append({
        'predicted_code': result['diagnosis']['code'],
        'predicted_name': result['diagnosis']['name'],
        'confidence': result['diagnosis']['confidence']
    })

# Create results dataframe
df_results = pd.DataFrame(results)
df = pd.concat([df, df_results], axis=1)
df.head()
```

### Batch Processing Script

```python
import json
from pathlib import Path
from final_inference import ShifaMindPredictor
from tqdm import tqdm

def batch_predict(input_dir, output_dir):
    predictor = ShifaMindPredictor()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Process all text files
    for file_path in tqdm(list(input_path.glob('*.txt'))):
        text = file_path.read_text()
        result = predictor.predict(text)

        # Save result
        output_file = output_path / f"{file_path.stem}_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

if __name__ == '__main__':
    batch_predict('/path/to/input', '/path/to/output')
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory

```python
# Use CPU instead of GPU
predictor = ShifaMindPredictor(device='cpu')

# Or reduce batch size
results = predictor.predict_batch(notes, batch_size=4)
```

#### 2. Slow Inference

```python
# Disable concept prediction
result = predictor.predict(text, return_concepts=False)
```

#### 3. Checkpoint Not Found

```python
# Specify full path
predictor = ShifaMindPredictor(
    checkpoint_path='/full/path/to/checkpoint.pt'
)
```

#### 4. Import Errors

```bash
# Ensure you're in the ShifaMind directory
cd /path/to/ShifaMind
python final_inference.py ...

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/ShifaMind:$PYTHONPATH
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from final_inference import ShifaMindPredictor
predictor = ShifaMindPredictor()
```

### Performance Optimization

```python
# Use GPU if available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Check GPU usage
import nvidia_smi
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print(f"GPU Memory: {info.used / 1e9:.2f} GB / {info.total / 1e9:.2f} GB")
```

---

## Advanced Usage

### Custom Filtering

```python
from final_concept_filter import get_filtered_top_concepts
import numpy as np

# Get raw concept scores
concept_scores = np.array([...])  # From model output

# Apply custom filtering
filtered_concepts = get_filtered_top_concepts(
    concept_scores,
    concept_store,
    top_k=15  # Custom number of concepts
)
```

### Knowledge Base Integration

```python
import json
from pathlib import Path

# Load knowledge base
kb_path = Path('03_Models/clinical_knowledge_base.json')
with open(kb_path) as f:
    knowledge_base = json.load(f)

# Retrieve knowledge for diagnosis
diagnosis_code = 'J189'
knowledge_entries = knowledge_base[diagnosis_code]

for entry in knowledge_entries[:3]:
    print(f"{entry['type']}: {entry['text']}")
```

---

## Next Steps

- See [SETUP.md](SETUP.md) for installation instructions
- See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- See [README.md](../README.md) for project overview

---

**For additional support, please refer to the main documentation or contact the author.**
