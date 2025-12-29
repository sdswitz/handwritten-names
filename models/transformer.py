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
    """Convert image to patch embeddings.

    Splits the input image into non-overlapping patches and projects each patch
    into an embedding vector using a linear layer.

    Args:
        img_height (int): Height of input image
        img_width (int): Width of input image
        patch_size (int): Size of each square patch
        embed_dim (int): Dimension of patch embeddings

    Example:
        For a 128x512 image with patch_size=64:
        - Number of patches: (128/64) × (512/64) = 2 × 8 = 16 patches
        - Each patch: 64×64 pixels = 4,096 values
        - After projection: 16 tokens, each embed_dim-dimensional
    """

    def __init__(self, img_height, img_width, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_h = img_height // patch_size
        self.num_patches_w = img_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size, embed_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (B, 1, H, W)

        Returns:
            torch.Tensor: Patch embeddings of shape (B, num_patches, embed_dim)
        """
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