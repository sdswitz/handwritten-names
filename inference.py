import os
import torch
from PIL import Image
from torchvision import transforms
import argparse

from config import Config
from models.crnn import CRNN
from utils.decoder import CTCDecoder
from utils.transforms import ResizePad


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    model = CRNN(
        img_height=Config.IMG_HEIGHT,
        img_width=Config.IMG_WIDTH,
        num_channels=Config.NUM_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        cnn_output_channels=Config.CNN_OUTPUT_CHANNELS,
        rnn_hidden_size=Config.RNN_HIDDEN_SIZE,
        rnn_num_layers=Config.RNN_NUM_LAYERS,
        rnn_dropout=Config.RNN_DROPOUT
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}: {e}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def preprocess_image(image_path):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        ResizePad((Config.IMG_HEIGHT, Config.IMG_WIDTH)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('L')
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def predict(model, image_path, decoder, device):
    """Predict text from image."""
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        output, output_lengths = model(image_tensor)

    # Decode prediction
    predictions = decoder.decode(output.cpu(), output_lengths.cpu())

    return predictions[0]


def main():
    parser = argparse.ArgumentParser(description='Predict handwritten name from image')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    args = parser.parse_args()

    # Set device
    device = Config.DEVICE
    print(f'Using device: {device}')

    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, device)

    # Create decoder
    decoder = CTCDecoder(Config.CHARS, Config.BLANK_LABEL)

    # Predict
    print(f'Predicting text from {args.image}...')
    prediction = predict(model, args.image, decoder, device)

    print('\n' + '=' * 50)
    print(f'Prediction: {prediction}')
    print('=' * 50)


if __name__ == '__main__':
    main()
