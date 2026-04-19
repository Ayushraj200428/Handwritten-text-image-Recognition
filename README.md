# Hindi Handwriting OCR

A deep learning system for recognizing handwritten Hindi (Devanagari script) text, built with a hybrid **CNN + Vision Transformer (ViT)** architecture and trained using **CTC loss**.

Live demo → deploy on [Streamlit Cloud](https://streamlit.io/cloud) using `app.py`.

---

## How It Works

### Architecture

```
Input Image (64×256 grayscale)
        ↓
CNN Feature Extractor        ← 3 conv blocks, extracts local stroke features
        ↓
Token Embedding              ← patch projection (Conv2d, patch_size=4)
        ↓
Positional Encoding          ← learnable position embeddings
        ↓
Vision Transformer Encoder   ← 6 transformer layers, 6 attention heads
        ↓
CTC Prediction Head          ← linear → 109 Devanagari character classes
        ↓
Greedy CTC Decode            → predicted Hindi text + confidence score
```

### Preprocessing Pipeline

Every image (upload or webcam) goes through the same pipeline used during training:

1. Convert to grayscale
2. CLAHE contrast enhancement (clipLimit=2.0)
3. Aspect-ratio preserving resize with white padding → 64×256
4. Normalize to [0, 1]

### CTC Decoding

- Greedy decode: argmax over timesteps
- Collapse consecutive identical tokens, remove blanks
- Confidence = `exp(mean log-prob of emitted non-blank characters)`

### Character Set

109 classes covering the full Devanagari Unicode block — vowels, consonants, matras, conjuncts, digits, punctuation, and a `<BLANK>` token for CTC.

---

## Project Structure

```
├── app.py                          # Streamlit web app
├── Major_Project.ipynb             # Training notebook
├── On_hindi_dataset_fixed.ipynb    # Dataset exploration
├── hindi_ocr_static.py             # Static dataset inference script
├── requirements.txt                # Python dependencies
├── NotoSansDevanagari-Regular.ttf  # Font for Devanagari rendering
└── README.md
```

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app expects `major_project_trained_model.keras` (PyTorch checkpoint) in the same directory. If not found, it runs with untrained weights and shows a warning.

---

## Deploying on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, set main file to `app.py`
4. Deploy

> **Note:** The trained model file (`major_project_trained_model.keras`) is a PyTorch checkpoint (~93 MB). GitHub blocks files >100 MB. If yours exceeds that, host it on [HuggingFace Hub](https://huggingface.co) or Google Drive and add a download snippet at the top of `app.py`:
> ```python
> import urllib.request
> if not Path("major_project_trained_model.keras").exists():
>     urllib.request.urlretrieve("YOUR_DIRECT_URL", "major_project_trained_model.keras")
> ```

---

## Requirements

```
streamlit>=1.35.0
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
pandas>=2.0.0
```

---

## Dataset

Trained on Hindi handwriting parquet datasets:
- `sikhna.parquet` — training set
- `pariksha.parquet` — test set

Each row contains a handwritten word image (bytes) and its ground truth text label.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Best epoch | 12 |
| Test CTC loss | 0.4250 |
| Parameters | ~24.5M (~93 MB) |
