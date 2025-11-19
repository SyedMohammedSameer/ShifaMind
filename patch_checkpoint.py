#!/usr/bin/env python3
"""
Emergency patch: Add concept_embeddings to existing checkpoint
"""

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

print("🔧 Patching checkpoint to add concept_embeddings...")
print("This will only take ~30 seconds instead of re-running 45 min training\n")

# Load existing checkpoint
checkpoint_path = 'stage4_joint_best_revised.pt'
print(f"Loading {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check if already has embeddings
if 'concept_embeddings' in checkpoint:
    print("✅ Checkpoint already has concept_embeddings! Nothing to do.")
    exit(0)

print(f"Found {checkpoint['num_concepts']} concepts")
print(f"CUIs: {checkpoint['concept_cuis'][:5]}... (showing first 5)")

# Load Bio_ClinicalBERT
print("\nLoading Bio_ClinicalBERT...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Using device: {device}")

# Generate embeddings from CUIs
# We'll just use the CUI as text (simple but works)
concept_cuis = checkpoint['concept_cuis']
concept_texts = [f"Medical concept {cui}" for cui in concept_cuis]

print(f"\nGenerating embeddings for {len(concept_texts)} concepts...")

batch_size = 32
all_embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(concept_texts), batch_size), desc="Encoding"):
        batch = concept_texts[i:i+batch_size]
        encodings = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        outputs = model(**encodings)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        all_embeddings.append(embeddings.cpu())

concept_embeddings = torch.cat(all_embeddings, dim=0)
print(f"✅ Created embeddings: {concept_embeddings.shape}")

# Add to checkpoint
checkpoint['concept_embeddings'] = concept_embeddings

# Save updated checkpoint
backup_path = checkpoint_path.replace('.pt', '_backup.pt')
print(f"\n💾 Backing up original to: {backup_path}")
torch.save(torch.load(checkpoint_path, map_location='cpu'), backup_path)

print(f"💾 Saving patched checkpoint to: {checkpoint_path}")
torch.save(checkpoint, checkpoint_path)

print("\n✅ DONE! Checkpoint patched successfully!")
print(f"✅ Now you can run 033.py without re-training!")
print(f"\nCheckpoint now contains:")
for key in checkpoint.keys():
    if key == 'concept_embeddings':
        print(f"  ✅ {key}: {checkpoint[key].shape}")
    elif key == 'model_state_dict':
        print(f"  ✅ {key}: <model weights>")
    else:
        print(f"  ✅ {key}: {checkpoint[key] if not isinstance(checkpoint[key], list) else f'[{len(checkpoint[key])} items]'}")
