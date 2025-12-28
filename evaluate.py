import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from config import Config
from data.dataset import HandwrittenNamesDataset, collate_fn
from models.crnn import CRNN
from utils.decoder import CTCDecoder
from utils.metrics import character_error_rate, word_error_rate, accuracy


def evaluate(model, dataloader, decoder, device):
    """Evaluate the model and return detailed results."""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')

        for images, labels, label_lengths, _ in pbar:
            images = images.to(device)

            # Forward pass
            output, output_lengths = model(images)

            # Decode predictions
            decoded_preds = decoder.decode(output.cpu(), output_lengths.cpu())

            # Reconstruct target texts
            label_list = labels.cpu().numpy()
            start_idx = 0
            for length in label_lengths.cpu().numpy():
                label_segment = label_list[start_idx:start_idx + length]
                target_text = ''.join([Config.CHARS[idx] for idx in label_segment])
                all_targets.append(target_text)
                start_idx += length

            all_predictions.extend(decoded_preds)

    # Calculate overall metrics
    cer = character_error_rate(all_predictions, all_targets)
    wer = word_error_rate(all_predictions, all_targets)
    acc = accuracy(all_predictions, all_targets)

    return all_predictions, all_targets, cer, wer, acc


def main():
    # Set device
    device = Config.DEVICE
    print(f'Using device: {device}')

    # Load test dataset
    print('Loading test dataset...')
    test_dataset = HandwrittenNamesDataset(
        csv_path=os.path.join(Config.DATA_DIR, Config.VAL_CSV),
        img_dir=os.path.join(Config.DATA_DIR, Config.VAL_IMG_DIR),
        img_height=Config.IMG_HEIGHT,
        img_width=Config.IMG_WIDTH,
        chars=Config.CHARS,
        augment=False
    )

    print(f'Test dataset size: {len(test_dataset)}')

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_fn
    )

    # Create model
    print('Creating model...')
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

    # Load checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f'Error: Checkpoint not found at {checkpoint_path}')
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')

    # Decoder
    decoder = CTCDecoder(Config.CHARS, Config.BLANK_LABEL)

    # Evaluate
    print('\nEvaluating model...')
    predictions, targets, cer, wer, acc = evaluate(model, test_loader, decoder, device)

    # Print results
    print('\n' + '=' * 50)
    print('EVALUATION RESULTS')
    print('=' * 50)
    print(f'Character Error Rate (CER): {cer:.4f}')
    print(f'Word Error Rate (WER): {wer:.4f}')
    print(f'Accuracy: {acc:.4f} ({acc*100:.2f}%)')
    print('=' * 50)

    # Save results
    results_df = pd.DataFrame({
        'target': targets,
        'prediction': predictions,
        'correct': [t == p for t, p in zip(targets, predictions)]
    })
    results_path = os.path.join(Config.CHECKPOINT_DIR, 'evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f'\nResults saved to {results_path}')

    # Show some examples
    print('\nSample predictions:')
    print('-' * 50)
    for i in range(min(20, len(predictions))):
        correct = '✓' if predictions[i] == targets[i] else '✗'
        print(f'{correct} Target: {targets[i]:20s} | Prediction: {predictions[i]:20s}')


if __name__ == '__main__':
    main()
