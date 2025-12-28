import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from utils.transforms import ResizePad
from config import Config

class HandwrittenNamesDataset(Dataset):
    """
    Dataset for handwritten names.

    Args:
        csv_path: Path to CSV file with filename and identity columns
        img_dir: Directory containing images
        img_height: Target image height
        img_width: Target image width
        chars: String of all characters in vocabulary
        augment: Whether to apply data augmentation
    """
    def __init__(self, csv_path, img_dir, img_height=128, img_width=512,
                 chars=' ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.chars = chars
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        self.augment = augment

        # Base transforms
        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            ResizePad((img_height, img_width)),
        ]

        # Add augmentation for training
        if augment:
            transform_list.insert(1, transforms.RandomApply([
                transforms.RandomRotation(degrees=5),
            ], p=0.3))
            transform_list.insert(2, transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3),
            ], p=0.2))

        transform_list.append(transforms.ToTensor())

        # Normalize to [-1, 1] or [0, 1] depending on preference
        # Using [0, 1] for now

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image filename and label
        row = self.df.iloc[idx]
        img_name = row['FILENAME']
        label_text = str(row['IDENTITY']).upper()  # Ensure uppercase

        # Load and transform image
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('L')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = torch.zeros(1, Config.IMG_HEIGHT, Config.IMG_WIDTH)
            label_text = ""

        # Encode label text to indices
        label = self.encode_text(label_text)
        label_length = len(label)

        return image, label, label_length

    def encode_text(self, text):
        """Convert text string to list of character indices."""
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                # Skip unknown characters
                pass
        return encoded

    def decode_indices(self, indices):
        """Convert list of indices back to text string."""
        text = ""
        for idx in indices:
            if idx < len(self.chars):
                text += self.idx_to_char[idx]
        return text


def collate_fn(batch):
    """
    Custom collate function to handle variable-length labels.

    Returns:
        images: Tensor of shape (batch_size, channels, height, width)
        labels: Concatenated labels as 1D tensor
        label_lengths: Length of each label
        input_lengths: Length of model output sequence for each sample
    """
    images, labels, label_lengths = zip(*batch)

    # Stack images
    images = torch.stack(images, dim=0)

    # Concatenate all labels into a single 1D tensor (required by CTCLoss)
    labels_concat = []
    for label in labels:
        labels_concat.extend(label)
    labels_tensor = torch.LongTensor(labels_concat)

    # Label lengths
    label_lengths = torch.LongTensor(label_lengths)

    # Input lengths - will be determined by model output sequence length
    # For now, placeholder - will be set during training based on actual model output
    # The CRNN will output a sequence based on image width / downsampling factor
    batch_size = len(images)
    input_lengths = torch.full((batch_size,), fill_value=64, dtype=torch.long)  # Placeholder

    return images, labels_tensor, label_lengths, input_lengths
