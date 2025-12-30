"""
Transformer-based OCR model for handwritten name recognition.

This module implements a Vision Transformer (ViT) style architecture that uses:
- Patch embeddings: Split image into patches, embed each as a token
- Transformer encoder: Self-attention to model relationships between patches
- CTC loss: Predict character sequence without explicit alignment

Architecture:
    Input Image (128 x 512)
        ↓
    Patchify (split into patches)
        ↓
    Patch Embeddings (linear projection)
        ↓
    Positional Encoding (add position info)
        ↓
    Transformer Encoder (multi-head self-attention)
        ↓
    Output Projection (to character probabilities)
        ↓
    CTC Decoding
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings using convolutional projection.

    Uses a convolutional layer with kernel_size=patch_size and stride=patch_size
    to efficiently extract and project patches. This is the standard approach used
    in Vision Transformers (ViT, DeiT, Swin, etc.).

    Args:
        img_height (int): Height of input image
        img_width (int): Width of input image
        patch_size (int): Size of each square patch
        in_chans (int): Number of input channels (default: 1 for grayscale)
        embed_dim (int): Dimension of patch embeddings
        bias (bool): Whether to use bias in the projection layer

    Example:
        For a 128x512 image with patch_size=16:
        - Number of patches: (128/16) × (512/16) = 8 × 32 = 256 patches
        - Each patch: 16×16 pixels processed by convolution
        - After projection: 256 tokens, each embed_dim-dimensional
    """

    def __init__(self, img_height, img_width, patch_size, in_chans=1, embed_dim=256, bias=True):
        super().__init__()

        assert img_height % patch_size == 0, f"Image height {img_height} must be divisible by patch_size {patch_size}"
        assert img_width % patch_size == 0, f"Image width {img_width} must be divisible by patch_size {patch_size}"

        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.num_patches_h = img_height // patch_size
        self.num_patches_w = img_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Convolutional projection (standard ViT approach)
        # Each "convolution" processes exactly one patch (kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            torch.Tensor: Patch embeddings of shape (B, num_patches, embed_dim)
        """
        # x: (B, 1, H, W)
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)

        return x


class PositionalEncoding(nn.Module):
    """Add positional information to patch embeddings.

    Uses sinusoidal positional encoding to inject information about the
    relative or absolute position of patches in the sequence.

    Args:
        num_patches (int): Total number of patches
        embed_dim (int): Dimension of embeddings
        dropout (float): Dropout probability
    """

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
        """
        Args:
            x (torch.Tensor): Patch embeddings of shape (B, num_patches, embed_dim)

        Returns:
            torch.Tensor: Positionally encoded embeddings
        """
        x = x + self.pe
        return self.dropout(x)


class TransformerOCR(nn.Module):
    """Transformer model for handwritten name recognition.

    This model uses a Vision Transformer (ViT) style architecture to recognize
    handwritten names from images. It splits images into patches, processes them
    through a transformer encoder, and predicts character sequences using CTC loss.

    Args:
        img_height (int): Height of input images (default: 128)
        img_width (int): Width of input images (default: 512)
        patch_size (int): Size of square patches (default: 64)
        embed_dim (int): Embedding dimension (default: 256)
        num_layers (int): Number of transformer encoder layers (default: 6)
        num_heads (int): Number of attention heads (default: 8)
        dim_ff (int): Dimension of feed-forward network (default: 1024)
        num_classes (int): Number of output classes (default: 38)
        dropout (float): Dropout probability (default: 0.1)

    Example:
        >>> model = TransformerOCR(
        ...     img_height=128, img_width=512, patch_size=64,
        ...     embed_dim=256, num_layers=6, num_heads=8
        ... )
        >>> x = torch.randn(4, 1, 128, 512)  # Batch of 4 images
        >>> output, output_lengths = model(x)
        >>> print(output.shape)  # (num_patches, batch_size, num_classes)
    """

    def __init__(self, img_height=128, img_width=512, patch_size=64,
                 embed_dim=256, num_layers=6, num_heads=8, dim_ff=1024,
                 num_classes=38, dropout=0.1, in_chans=1):
        super().__init__()

        # Patch embedding (using convolutional projection)
        self.patch_embed = PatchEmbedding(
            img_height=img_height,
            img_width=img_width,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
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
        """
        Args:
            x (torch.Tensor): Input images of shape (B, 1, H, W)

        Returns:
            tuple: (output, output_lengths)
                - output (torch.Tensor): Log probabilities of shape
                  (num_patches, B, num_classes) for CTC loss
                - output_lengths (torch.Tensor): Length of each sequence (all equal
                  to num_patches for this architecture)
        """
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

        # Output lengths (all sequences have same length = num_patches)
        batch_size = x.size(0)
        output_lengths = torch.full(
            (batch_size,),
            fill_value=output.size(0),
            dtype=torch.long
        )

        return output, output_lengths


if __name__ == '__main__':
    # Test the model
    print("Testing TransformerOCR model...")
    print("-" * 50)

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

    # Create sample input
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 512)

    # Forward pass
    output, output_lengths = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output lengths: {output_lengths}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 50)

    # Calculate model size
    num_patches_h = 128 // 64
    num_patches_w = 512 // 64
    num_patches = num_patches_h * num_patches_w
    print(f"\nModel architecture details:")
    print(f"  Image size: 128 x 512")
    print(f"  Patch size: 64 x 64")
    print(f"  Number of patches: {num_patches_h} x {num_patches_w} = {num_patches}")
    print(f"  Embedding dimension: 256")
    print(f"  Transformer layers: 6")
    print(f"  Attention heads: 8")
    print(f"  Feed-forward dimension: 1024")
    print(f"  Output classes: 38")
    print("\nTest completed successfully!")