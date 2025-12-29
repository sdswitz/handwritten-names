# Claude Development Notes

This document contains development notes and context from building this handwritten name recognition system.

## Project Overview

Built a CRNN (Convolutional Recurrent Neural Network) model to recognize handwritten names from images. The model performs character-level predictions using CTC (Connectionist Temporal Classification) loss.

**Goal:** Create a model that can read handwritten names from images, with eventual plans to build a web interface where users can draw their names and get predictions.

## Dataset

- **Source:** Kaggle dataset `landlord/handwriting-recognition`
- **Size:** 330,961 training images, 41,370 validation images
- **Format:** Grayscale images of varying sizes (max ~388x72 pixels)
- **Labels:** Names in uppercase (A-Z, space, digits 0-9)
- **CSVs:** FILENAME, IDENTITY columns mapping images to text labels

## How the Model Works

### Architecture: CRNN (CNN + RNN + CTC)

The model processes images in three stages:

1. **CNN (Feature Extraction)**
   - Input: 128x512 grayscale images
   - 5 convolutional blocks with batch normalization
   - Reduces height dimension while preserving width
   - Output: Feature maps with shape (batch, 512, 7, 127)

2. **RNN (Sequence Modeling)**
   - Reshape CNN output: (batch, width, features*height)
   - 2-layer bidirectional LSTM (256 hidden units per direction)
   - Models sequential dependencies left-to-right
   - Each position in width represents a "time step"

3. **CTC (Decoding)**
   - Linear layer maps LSTM output to character probabilities
   - CTC loss handles variable-length outputs without alignment
   - Decoding: argmax at each time step → collapse repeats → remove blanks

### Character-Level Prediction

The model **does NOT** predict whole names. Instead:

- Each horizontal position in the image gets a character prediction
- Example output for "HELLO":
  ```
  Time:  0  1  2  3  4  5  6  7  8  9  10 ...
  Pred:  - - H H E E L L L O  O  - ...
  ```
- CTC decoder:
  1. Removes repeated characters: H E L O
  2. Removes blanks (-): HELLO

This allows variable-length names without needing to know character boundaries.

## Project Structure

```
handwritten-names/
├── config.py              # All hyperparameters and paths
├── train.py               # Training loop with validation
├── evaluate.py            # Model evaluation script
├── inference.py           # Single image prediction
├── colab_training.ipynb   # Google Colab training notebook
├── requirements.txt       # Python dependencies
├── README.md             # User documentation
├── CLAUDE.md             # This file
│
├── data/
│   ├── dataset.py        # Custom Dataset class
│   └── __init__.py       # Encodes text to indices, handles variable lengths
│
├── models/
│   ├── crnn.py           # CRNN architecture (~15M parameters)
│   └── __init__.py
│
└── utils/
    ├── transforms.py     # ResizePad (aspect-ratio preserving resize)
    ├── decoder.py        # CTC greedy decoder
    ├── metrics.py        # CER, WER, Accuracy
    └── __init__.py
```

## Key Implementation Details

### 1. Data Preprocessing
- **ResizePad transform:** Maintains aspect ratio, pads to 128x512
- **Augmentation (training only):**
  - Random rotation ±5° (30% probability)
  - Gaussian blur (20% probability)
- **Normalization:** Images scaled to [0, 1] range

### 2. Character Vocabulary
```python
CHARS = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # 37 characters
BLANK_LABEL = 37  # CTC blank token
NUM_CLASSES = 38  # Total classes
```

### 3. Training Configuration
- **Batch size:** 32
- **Learning rate:** 0.001 with ReduceLROnPlateau scheduler
- **Optimizer:** Adam with weight decay 1e-5
- **Early stopping:** Patience of 5 epochs based on validation CER
- **Checkpointing:** Saves best model + periodic checkpoints every 5 epochs

### 4. Evaluation Metrics
- **CER (Character Error Rate):** Edit distance / total characters
- **WER (Word Error Rate):** Proportion of incorrect predictions
- **Accuracy:** Exact match percentage

## Training Workflow

### Local Training (Not Recommended - CPU too slow)
```bash
python train.py
```

### Google Colab Training (Recommended)
1. Open `colab_training.ipynb` in Google Colab
2. Set runtime to GPU (Runtime → Change runtime type → T4 GPU)
3. Upload `kaggle.json` API key
4. Run cells sequentially:
   - Dataset downloads from Kaggle in ~30 seconds
   - Training takes 1-3 hours on T4 GPU
   - Checkpoints auto-save to Google Drive

**Why Colab?**
- Training on CPU would take days/weeks
- Colab provides free GPU (T4 or A100)
- Dataset downloads directly from Kaggle (no manual upload needed)
- Checkpoints saved to Google Drive (persistent storage)

## Important Design Decisions

### 1. Image Size (128x512)
- Height=128: Good balance for name images (original max ~72)
- Width=512: Accommodates long names while keeping memory manageable
- Aspect ratio preserved with padding (no distortion)

### 2. Why CRNN?
- Standard architecture for sequence recognition tasks (OCR)
- CNN: Efficient feature extraction from images
- RNN: Captures sequential nature of text
- CTC: No need for character-level annotations

### 3. Why Greedy Decoding (not Beam Search)?
- Simpler implementation for initial version
- Beam search can be added later for better accuracy
- Greedy decoding is fast and usually sufficient

### 4. Data Storage Strategy
- **Code:** GitHub repository
- **Dataset:** Downloaded from Kaggle (not in git)
- **Checkpoints:** Google Drive (gitignored - too large for GitHub)
- **Notebooks:** Only `colab_training.ipynb` in git (EDA notebooks excluded)

## Troubleshooting

### Issue 1: ReduceLROnPlateau verbose parameter
**Error:** `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`

**Fix:** Removed `verbose=True` from scheduler initialization (removed in newer PyTorch versions)

### Issue 2: Slow dataset upload to Google Drive
**Solution:** Changed to Kaggle API direct download in Colab
- Downloads entire dataset in ~30 seconds
- No need to manually upload 50k+ images

## Model Performance Expectations

Based on similar CRNN implementations:
- **Expected CER:** 5-15% (lower is better)
- **Expected Accuracy:** 70-85% (exact match)
- **Training time:** 1-3 hours on T4 GPU for 50 epochs

Performance depends on:
- Dataset quality (handwriting clarity)
- Model convergence
- Hyperparameter tuning

## Future Improvements

### Short-term
1. **Beam search decoding:** Better predictions than greedy
2. **Learning rate scheduling:** Fine-tune schedule for faster convergence
3. **Data augmentation:** Add elastic transforms, perspective shifts
4. **Experiment tracking:** Add wandb/tensorboard logging

### Medium-term
1. **Language model integration:** Use word probabilities to improve predictions
2. **Attention mechanism:** Replace/augment LSTM with attention
3. **Transfer learning:** Use pretrained CNN backbone (ResNet, EfficientNet)
4. **Mixed precision training:** Faster training with fp16

### Long-term
1. **Web interface:** Flask/Streamlit app for drawing names
2. **Real-time inference:** Optimize for <100ms latency
3. **Mobile deployment:** Convert to ONNX/TFLite for mobile apps
4. **Multi-language support:** Extend beyond English names

## GitHub Repository Structure

- **Main branch:** Contains all production code
- **Ignored files (.gitignore):**
  - Data files (CSVs, images)
  - Model checkpoints (.pth)
  - Jupyter notebooks (except colab_training.ipynb)
  - Python cache, IDE files, logs

## Notes for Future Sessions

1. **Model weights:** After training, download `best_model.pth` from Google Drive
2. **Character vocab:** Hard-coded in config.py - verify it matches your data
3. **Inference:** Use `inference.py` for single image predictions
4. **Evaluation:** Run `evaluate.py` on validation set for detailed results
5. **Config changes:** All hyperparameters centralized in `config.py`

## Resources

- **Kaggle Dataset:** https://www.kaggle.com/datasets/landlord/handwriting-recognition
- **GitHub Repo:** https://github.com/sdswitz/handwritten-names
- **PyTorch CTCLoss Docs:** https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
- **CRNN Paper:** "An End-to-End Trainable Neural Network for Image-based Sequence Recognition"

## Development Timeline

1. ✅ Initial exploration in Jupyter notebook (EDA)
2. ✅ Built complete CRNN model architecture
3. ✅ Implemented dataset loader with text encoding
4. ✅ Created training script with metrics tracking
5. ✅ Added evaluation and inference scripts
6. ✅ Created Google Colab notebook with Kaggle integration
7. ✅ Fixed PyTorch compatibility issues
8. 🔄 **Current:** Ready to train model in Colab
9. ⏭️ **Next:** Web interface for interactive predictions

---

*This file is meant to provide context for future development sessions with Claude or other developers.*
