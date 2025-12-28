# Handwritten Names Recognition

A deep learning project to recognize handwritten names using a CRNN (Convolutional Recurrent Neural Network) architecture with CTC (Connectionist Temporal Classification) loss.

## Architecture

The model consists of three main components:

1. **CNN (Convolutional Neural Network)**: Extracts visual features from images
2. **RNN (Recurrent Neural Network)**: Bidirectional LSTM that models sequential dependencies in the features
3. **CTC Loss**: Handles variable-length outputs without requiring character-level alignment

### How It Works

- Images are processed through CNN layers that reduce height while preserving width
- The CNN output is treated as a sequence of features (left to right)
- The LSTM processes this sequence and outputs character probabilities at each time step
- CTC decoding collapses repeated characters and removes blanks to produce the final text

## Project Structure

```
handwritten-names/
├── data/
│   ├── __init__.py
│   └── dataset.py          # Custom Dataset class for loading images and labels
├── models/
│   ├── __init__.py
│   └── crnn.py            # CRNN model architecture
├── utils/
│   ├── __init__.py
│   ├── transforms.py      # Image preprocessing (ResizePad)
│   ├── decoder.py         # CTC decoder
│   └── metrics.py         # Evaluation metrics (CER, WER, Accuracy)
├── config.py              # Configuration and hyperparameters
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── inference.py           # Single image inference
└── requirements.txt
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data is organized as:
```
/path/to/data/
├── train_v2/train/        # Training images
├── validation_v2/validation/  # Validation images
├── test_v2/test/          # Test images
├── written_name_train_v2.csv
├── written_name_validation_v2.csv
└── written_name_test_v2.csv
```

3. Update the `DATA_DIR` in [config.py](config.py) to point to your data directory.

## Training

To train the model:

```bash
python train.py
```

Training parameters can be adjusted in [config.py](config.py):
- `BATCH_SIZE`: Batch size for training (default: 32)
- `NUM_EPOCHS`: Number of training epochs (default: 50)
- `LEARNING_RATE`: Initial learning rate (default: 0.001)
- `IMG_HEIGHT`, `IMG_WIDTH`: Input image dimensions (default: 128x512)

The training script will:
- Save the best model based on validation CER to `checkpoints/best_model.pth`
- Save periodic checkpoints
- Log training history to `checkpoints/training_history.csv`
- Use early stopping if validation CER doesn't improve

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py
```

This will:
- Load the best checkpoint
- Evaluate on the validation set
- Print CER, WER, and accuracy
- Save detailed results to `checkpoints/evaluation_results.csv`

## Inference

To predict a name from a single image:

```bash
python inference.py --image path/to/image.jpg
```

Optional arguments:
- `--checkpoint`: Path to model checkpoint (default: `checkpoints/best_model.pth`)

## Metrics

The model is evaluated using three metrics:

- **Character Error Rate (CER)**: Edit distance at character level normalized by target length
- **Word Error Rate (WER)**: Proportion of incorrect predictions (binary for single words)
- **Accuracy**: Proportion of exact matches

## Model Details

Default configuration:
- Input: 128x512 grayscale images
- Character vocabulary: uppercase A-Z, space, digits 0-9 (38 classes total)
- CNN: 5 convolutional blocks with batch normalization
- RNN: 2-layer bidirectional LSTM with 256 hidden units
- Total parameters: ~20M (varies based on config)

## Data Augmentation

Training uses:
- Random rotation (±5 degrees, 30% probability)
- Gaussian blur (20% probability)
- Aspect-ratio preserving resize with padding

## Future Improvements

- Beam search decoding for better predictions
- Language model integration
- Attention mechanism
- Transfer learning from pretrained vision models
- Web interface for interactive testing
