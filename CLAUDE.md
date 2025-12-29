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

## Alternative Architecture: Transformer with Patch Embeddings

### Overview

A Vision Transformer (ViT) style approach for handwritten name recognition. Instead of CNN+RNN, this uses:
- **Patch embeddings:** Split image into patches, embed each as a token
- **Transformer encoder:** Self-attention to model relationships between patches
- **Sequence decoder:** Predict character sequence autoregressively or with CTC

### How It Differs from CRNN

| Aspect | CRNN | Transformer |
|--------|------|-------------|
| Feature extraction | CNN (inductive bias for images) | Patch embeddings (learns from scratch) |
| Sequence modeling | Bidirectional LSTM | Self-attention (parallel processing) |
| Receptive field | Local (conv filters) | Global (attention over all patches) |
| Training | Faster convergence | Needs more data/compute |
| Inference | Sequential RNN processing | Fully parallel (faster) |

### Architecture Breakdown

```
Input Image (128 x 512)
    ↓
Patchify (split into 8x8 patches)
    ↓
Patch Embeddings (2 x 64 = 128 patches, each 64-dim)
    ↓
Positional Encoding (add position info)
    ↓
Transformer Encoder (6-12 layers)
    - Multi-head self-attention
    - Feed-forward networks
    - Layer normalization
    ↓
Sequence Prediction (two options):
  A) CTC Loss (like CRNN)
  B) Autoregressive decoder (like GPT)
    ↓
Output: Character sequence
```

### Patch Embeddings Explained

**Concept:** Treat an image like a sequence of "visual words" (patches).

For a 128x512 image with patch_size=64:
- Number of patches: (128/64) × (512/64) = 2 × 8 = 16 patches
- Each patch: 64×64 pixels = 4,096 values
- Linear projection: 4,096 → embed_dim (e.g., 256)
- Result: 16 tokens, each 256-dim

**Code example:**
```python
# Patchify the image
patch_size = 64
num_patches_h = 128 // patch_size  # 2
num_patches_w = 512 // patch_size  # 8

# Reshape: (B, 1, 128, 512) → (B, num_patches, patch_size²)
x = x.reshape(B, 1, num_patches_h, patch_size, num_patches_w, patch_size)
x = x.permute(0, 2, 4, 1, 3, 5)  # (B, 2, 8, 1, 64, 64)
x = x.reshape(B, num_patches_h * num_patches_w, -1)  # (B, 16, 4096)

# Linear embedding
patch_embed = nn.Linear(patch_size * patch_size, embed_dim)
x = patch_embed(x)  # (B, 16, 256)
```

### What Can Be Reused

From your current implementation:

✅ **Keep as-is:**
- `data/dataset.py` - Dataset and text encoding
- `utils/transforms.py` - ResizePad transform
- `utils/decoder.py` - CTC decoder (if using CTC loss)
- `utils/metrics.py` - Evaluation metrics
- `train.py` - Training loop structure (modify model instantiation)
- `config.py` - Configuration (add transformer hyperparameters)

🔄 **Modify:**
- `models/crnn.py` → Create new `models/transformer.py`
- Update Config with transformer-specific params

### Getting Started: Step-by-Step

#### Step 1: Add Transformer Config

Edit `config.py`:
```python
# Add after CRNN settings:

# Transformer architecture (alternative to CRNN)
USE_TRANSFORMER = False  # Toggle between CRNN and Transformer

# Patch embedding settings
PATCH_SIZE = 64  # Size of each patch (64x64)
EMBED_DIM = 256  # Embedding dimension

# Transformer settings
TRANSFORMER_LAYERS = 6  # Number of encoder layers
TRANSFORMER_HEADS = 8   # Number of attention heads
TRANSFORMER_DIM_FF = 1024  # Feed-forward dimension
TRANSFORMER_DROPOUT = 0.1
```

#### Step 2: Create Transformer Model

Create `models/transformer.py`:
```python
import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""
    def __init__(self, img_height, img_width, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_h = img_height // patch_size
        self.num_patches_w = img_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size, embed_dim)

    def forward(self, x):
        # x: (B, 1, H, W)
        B = x.shape[0]

        # Patchify: (B, 1, H, W) -> (B, num_patches, patch_size²)
        x = x.reshape(
            B, 1,
            self.num_patches_h, self.patch_size,
            self.num_patches_w, self.patch_size
        )
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, nH, nW, 1, pH, pW)
        x = x.reshape(B, self.num_patches, -1)  # (B, num_patches, patch_size²)

        # Linear projection
        x = self.projection(x)  # (B, num_patches, embed_dim)

        return x


class PositionalEncoding(nn.Module):
    """Add positional information to patches."""
    def __init__(self, num_patches, embed_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding
        pe = torch.zeros(1, num_patches, embed_dim)
        position = torch.arange(0, num_patches).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            -(math.log(10000.0) / embed_dim)
        )

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class TransformerOCR(nn.Module):
    """Transformer model for handwritten name recognition."""
    def __init__(self, img_height=128, img_width=512, patch_size=64,
                 embed_dim=256, num_layers=6, num_heads=8, dim_ff=1024,
                 num_classes=38, dropout=0.1):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_height, img_width, patch_size, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Positional encoding
        self.pos_encoding = PositionalEncoding(num_patches, embed_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection for CTC
        self.output_proj = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, H, W)

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)  # (B, num_patches, embed_dim)

        # Project to character probabilities
        output = self.output_proj(x)  # (B, num_patches, num_classes)

        # Permute for CTC loss: (num_patches, B, num_classes)
        output = output.permute(1, 0, 2)

        # Log softmax
        output = torch.nn.functional.log_softmax(output, dim=2)

        # Output lengths
        batch_size = x.size(0)
        output_lengths = torch.full(
            (batch_size,),
            fill_value=output.size(0),
            dtype=torch.long
        )

        return output, output_lengths


if __name__ == '__main__':
    # Test the model
    model = TransformerOCR(
        img_height=128,
        img_width=512,
        patch_size=64,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        dim_ff=1024,
        num_classes=38
    )

    x = torch.randn(4, 1, 128, 512)
    output, output_lengths = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output lengths: {output_lengths}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
```

#### Step 3: Modify Training Script

In `train.py`, add model selection:
```python
from models.crnn import CRNN
from models.transformer import TransformerOCR

# In main():
if Config.USE_TRANSFORMER:
    print('Creating Transformer model...')
    model = TransformerOCR(
        img_height=Config.IMG_HEIGHT,
        img_width=Config.IMG_WIDTH,
        patch_size=Config.PATCH_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_layers=Config.TRANSFORMER_LAYERS,
        num_heads=Config.TRANSFORMER_HEADS,
        dim_ff=Config.TRANSFORMER_DIM_FF,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.TRANSFORMER_DROPOUT
    )
else:
    print('Creating CRNN model...')
    model = CRNN(...)  # existing code
```

#### Step 4: Update Inference

Modify `inference.py` to load the appropriate model type:
```python
def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check which model type
    if Config.USE_TRANSFORMER:
        model = TransformerOCR(...)
    else:
        model = CRNN(...)

    model.load_state_dict(checkpoint['model_state_dict'])
    # ... rest of code
```

#### Step 5: Train and Compare

```bash
# Train CRNN (baseline)
# In config.py: USE_TRANSFORMER = False
python train.py

# Train Transformer
# In config.py: USE_TRANSFORMER = True
python train.py

# Compare results
python evaluate.py  # Check CER, WER, accuracy
```

### Expected Benefits

✅ **Advantages:**
1. **Global context:** Attention sees entire image at once
2. **Parallelization:** Faster training than sequential RNN
3. **State-of-the-art:** Transformers dominate many vision tasks
4. **Flexibility:** Easy to add position-aware attention
5. **Scalability:** Performance improves with more data/compute

⚠️ **Challenges:**
1. **Data hungry:** Needs more training data than CRNN
2. **Compute intensive:** Larger model, more memory
3. **Hyperparameter tuning:** More knobs to tune
4. **Patch size:** Critical choice (too small = too many tokens, too large = loss of detail)

### Optimization Tips

1. **Smaller patches for better detail:** Try 32×32 or 16×16 patches
2. **Pre-training:** Use ImageNet pretrained ViT and fine-tune
3. **Hierarchical patches:** Different patch sizes for multi-scale features
4. **Learnable positional embeddings:** Instead of fixed sinusoidal
5. **Cross-attention decoder:** Instead of CTC, use transformer decoder

### When to Use Transformer vs CRNN

**Use CRNN if:**
- Limited compute/data
- Need fast prototyping
- Simpler is better
- Known to work well for OCR

**Use Transformer if:**
- Have large dataset (100k+ samples)
- GPU resources available
- Want state-of-the-art performance
- Experimenting with architectures

### Further Reading

- **Vision Transformer (ViT) Paper:** "An Image is Worth 16x16 Words"
- **TrOCR Paper:** "Transformer-based OCR with Pre-trained Models"
- **Attention Is All You Need:** Original transformer paper
- **PyTorch Vision Transformer:** timm library implementation

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
