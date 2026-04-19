# Handwriting Text Recognition 

A deep learning project for recognizing handwritten Hindi (Devanagari script) using a hybrid **CNN + Vision Transformer (ViT)** architecture trained with **CTC loss**.

---

## Files in this Repository

| File | Description |
|------|-------------|
| `Major_Project.ipynb` | Main notebook — model definition, training loop, and inference |
| `On_hindi_dataset_fixed.ipynb` | Dataset exploration and preprocessing analysis |
| `NotoSansDevanagari-Regular.ttf` | Font file used to render Devanagari text in output overlays |
| `.gitignore` | Excludes datasets, model weights, images, and cache files |

---

## Project Overview

This project recognizes handwritten Hindi words from images. It uses a hybrid deep learning architecture combining a CNN for local feature extraction and a Vision Transformer (ViT) for sequence modeling, trained end-to-end with CTC loss.

### Architecture

```
Input Image (64 × 256 grayscale)
        ↓
CNN Feature Extractor
  └─ 3 convolutional blocks (Conv → BN → ReLU → MaxPool)
        ↓
Token Embedding
  └─ Patch projection via Conv2d (patch_size = 4)
        ↓
Positional Encoding
  └─ Learnable position embeddings
        ↓
Vision Transformer Encoder
  └─ 6 transformer layers, 6 attention heads
        ↓
CTC Prediction Head
  └─ Linear → 109 Devanagari character classes
        ↓
Greedy CTC Decode → predicted Hindi text + confidence score
```

### Preprocessing Pipeline

Every image goes through the same pipeline during both training and inference:

1. Convert to grayscale
2. CLAHE contrast enhancement (`clipLimit=2.0`)
3. Aspect-ratio preserving resize with white padding → `64 × 256`
4. Normalize pixel values to `[0, 1]`

### CTC Decoding

- Greedy decode: argmax over all timesteps
- Collapse consecutive identical tokens, then remove blank tokens
- Confidence score = `exp(mean log-probability of emitted non-blank characters)`

### Character Set

109 classes covering the full Devanagari Unicode block:
- Vowels, consonants, matras, conjuncts
- Digits (०–९)
- Punctuation (। ॥)
- `<BLANK>` token for CTC

---

## How to Run

### 1. Install dependencies

```bash
pip install torch torchvision timm opencv-python-headless numpy pillow pandas tqdm
```

### 2. Open the notebook

```bash
jupyter notebook Major_Project.ipynb
```

### 3. Choose a mode when prompted

```
1. Train from scratch
2. Load model + run inference
3. Train then run inference
```

> The notebook expects `sikhna.parquet` (train) and `pariksha.parquet` (test) datasets, and saves the trained model as `major_project_trained_model.keras` (PyTorch checkpoint despite the extension).

---

## Model Details

| Property | Value |
|----------|-------|
| Parameters | ~24.5 million |
| Model size | ~93 MB |
| Input size | 64 × 256 grayscale |
| Output classes | 109 Devanagari characters |
| Best epoch | 12 |
| Best test CTC loss | 0.4250 |
