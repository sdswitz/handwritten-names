import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from config import Config
from data.dataset import HandwrittenNamesDataset, collate_fn
from models.crnn import CRNN
from utils.decoder import CTCDecoder
from utils.metrics import character_error_rate, word_error_rate, accuracy, AverageMeter


def train_epoch(model, dataloader, criterion, optimizer, decoder, device, epoch):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    cer_meter = AverageMeter()
    wer_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, (images, labels, label_lengths, _) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        # Forward pass
        output, output_lengths = model(images)
        output_lengths = output_lengths.to(device)

        # Calculate CTC loss
        loss = criterion(output, labels, output_lengths, label_lengths)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # Decode predictions for metrics
        with torch.no_grad():
            decoded_preds = decoder.decode(output.cpu(), output_lengths.cpu())

            # Reconstruct target texts from labels
            target_texts = []
            label_list = labels.cpu().numpy()
            start_idx = 0
            for length in label_lengths.cpu().numpy():
                label_segment = label_list[start_idx:start_idx + length]
                target_text = ''.join([Config.CHARS[idx] for idx in label_segment])
                target_texts.append(target_text)
                start_idx += length

            # Calculate metrics
            batch_cer = character_error_rate(decoded_preds, target_texts)
            batch_wer = word_error_rate(decoded_preds, target_texts)
            batch_acc = accuracy(decoded_preds, target_texts)

        # Update meters
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        cer_meter.update(batch_cer, batch_size)
        wer_meter.update(batch_wer, batch_size)
        acc_meter.update(batch_acc, batch_size)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'cer': f'{cer_meter.avg:.4f}',
            'wer': f'{wer_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })

    return loss_meter.avg, cer_meter.avg, wer_meter.avg, acc_meter.avg


def validate(model, dataloader, criterion, decoder, device):
    """Validate the model."""
    model.eval()

    loss_meter = AverageMeter()
    cer_meter = AverageMeter()
    wer_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')

        for images, labels, label_lengths, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            # Forward pass
            output, output_lengths = model(images)
            output_lengths = output_lengths.to(device)

            # Calculate loss
            loss = criterion(output, labels, output_lengths, label_lengths)

            # Decode predictions
            decoded_preds = decoder.decode(output.cpu(), output_lengths.cpu())

            # Reconstruct target texts
            target_texts = []
            label_list = labels.cpu().numpy()
            start_idx = 0
            for length in label_lengths.cpu().numpy():
                label_segment = label_list[start_idx:start_idx + length]
                target_text = ''.join([Config.CHARS[idx] for idx in label_segment])
                target_texts.append(target_text)
                start_idx += length

            # Calculate metrics
            batch_cer = character_error_rate(decoded_preds, target_texts)
            batch_wer = word_error_rate(decoded_preds, target_texts)
            batch_acc = accuracy(decoded_preds, target_texts)

            # Update meters
            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            cer_meter.update(batch_cer, batch_size)
            wer_meter.update(batch_wer, batch_size)
            acc_meter.update(batch_acc, batch_size)

            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'cer': f'{cer_meter.avg:.4f}',
                'wer': f'{wer_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })

    return loss_meter.avg, cer_meter.avg, wer_meter.avg, acc_meter.avg


def main():
    # Create checkpoint directory
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set device
    device = Config.DEVICE
    print(f'Using device: {device}')

    # Create datasets
    print('Loading datasets...')
    train_dataset = HandwrittenNamesDataset(
        csv_path=os.path.join(Config.DATA_DIR, Config.TRAIN_CSV),
        img_dir=os.path.join(Config.DATA_DIR, Config.TRAIN_IMG_DIR),
        img_height=Config.IMG_HEIGHT,
        img_width=Config.IMG_WIDTH,
        chars=Config.CHARS,
        augment=True
    )

    val_dataset = HandwrittenNamesDataset(
        csv_path=os.path.join(Config.DATA_DIR, Config.VAL_CSV),
        img_dir=os.path.join(Config.DATA_DIR, Config.VAL_IMG_DIR),
        img_height=Config.IMG_HEIGHT,
        img_width=Config.IMG_WIDTH,
        chars=Config.CHARS,
        augment=False
    )

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
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
    model = model.to(device)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=Config.BLANK_LABEL, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Decoder
    decoder = CTCDecoder(Config.CHARS, Config.BLANK_LABEL)

    # Training loop
    best_val_cer = float('inf')
    patience_counter = 0

    history = {
        'train_loss': [], 'train_cer': [], 'train_wer': [], 'train_acc': [],
        'val_loss': [], 'val_cer': [], 'val_wer': [], 'val_acc': []
    }

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f'\nEpoch {epoch}/{Config.NUM_EPOCHS}')

        # Train
        train_loss, train_cer, train_wer, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, decoder, device, epoch
        )

        # Validate
        val_loss, val_cer, val_wer, val_acc = validate(
            model, val_loader, criterion, decoder, device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_cer'].append(train_cer)
        history['train_wer'].append(train_wer)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_cer'].append(val_cer)
        history['val_wer'].append(val_wer)
        history['val_acc'].append(val_acc)

        # Print epoch summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'Train - Loss: {train_loss:.4f}, CER: {train_cer:.4f}, WER: {train_wer:.4f}, Acc: {train_acc:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, CER: {val_cer:.4f}, WER: {val_wer:.4f}, Acc: {val_acc:.4f}')

        # Save best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            patience_counter = 0
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f'Saved best model with CER: {val_cer:.4f}')
        else:
            patience_counter += 1

        # Save checkpoint every N epochs
        if epoch % Config.SAVE_FREQ == 0:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cer': val_cer,
                'val_acc': val_acc,
            }, checkpoint_path)

        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f'\nEarly stopping triggered after {epoch} epochs')
            break

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(Config.CHECKPOINT_DIR, 'training_history.csv'), index=False)
    print('\nTraining complete!')
    print(f'Best validation CER: {best_val_cer:.4f}')


if __name__ == '__main__':
    main()
