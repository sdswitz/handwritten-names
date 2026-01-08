import torch
from models.ViT import ViT
from config_overfit_smallbatch import Config

def count_parameters(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def print_parameter_breakdown(model):
    """Print parameter count for each component of the model."""

    total_params = count_parameters(model)

    print("=" * 70)
    print("ViT Model Parameter Breakdown")
    print("=" * 70)

    # Patch Embedding
    patch_embed_params = count_parameters(model.patch_embed)
    print(f"\n{'Patch Embedding':<30} {patch_embed_params:>12,} params ({patch_embed_params/total_params*100:>5.2f}%)")
    print(f"  └─ Conv2d projection          {count_parameters(model.patch_embed.proj):>12,}")

    # Positional Embeddings
    pos_embed_params = model.pos_embed.numel()
    print(f"\n{'Positional Embeddings':<30} {pos_embed_params:>12,} params ({pos_embed_params/total_params*100:>5.2f}%)")
    print(f"  └─ Shape: {tuple(model.pos_embed.shape)}")

    # Transformer Blocks
    print(f"\n{'Transformer Blocks':<30} (x{len(model.blocks)} layers)")
    total_block_params = sum(count_parameters(block) for block in model.blocks)
    print(f"  Total                         {total_block_params:>12,} params ({total_block_params/total_params*100:>5.2f}%)")

    # Breakdown of a single block
    if len(model.blocks) > 0:
        single_block = model.blocks[0]
        single_block_params = count_parameters(single_block)
        print(f"  └─ Per block                  {single_block_params:>12,} params")

        # Attention
        attn_params = count_parameters(single_block.attn)
        print(f"     ├─ Self-Attention          {attn_params:>12,}")
        print(f"     │  ├─ QKV projection       {count_parameters(single_block.attn.qkv):>12,}")
        print(f"     │  └─ Output projection    {count_parameters(single_block.attn.proj):>12,}")

        # MLP
        mlp_params = count_parameters(single_block.mlp)
        print(f"     ├─ MLP                     {mlp_params:>12,}")
        print(f"     │  ├─ FC1                  {count_parameters(single_block.mlp.fc1):>12,}")
        print(f"     │  └─ FC2                  {count_parameters(single_block.mlp.fc2):>12,}")

        # Layer Norms
        norm_params = count_parameters(single_block.norm1) + count_parameters(single_block.norm2)
        print(f"     └─ Layer Norms (x2)        {norm_params:>12,}")

    # Output Head
    print(f"\n{'Output Head':<30}")
    norm_params = count_parameters(model.norm)
    head_params = count_parameters(model.head)
    print(f"  ├─ Layer Norm                 {norm_params:>12,} params ({norm_params/total_params*100:>5.2f}%)")
    print(f"  └─ Linear (Classification)    {head_params:>12,} params ({head_params/total_params*100:>5.2f}%)")

    print("\n" + "=" * 70)
    print(f"{'TOTAL PARAMETERS':<30} {total_params:>12,}")
    print("=" * 70)

    # Memory estimate
    memory_mb = total_params * 4 / (1024**2)  # 4 bytes per float32
    print(f"\nEstimated model size (float32): {memory_mb:.2f} MB")
    print(f"Estimated model size (float16): {memory_mb/2:.2f} MB")

if __name__ == '__main__':
    # Create model with config settings
    model = ViT(
        img_h=Config.IMG_HEIGHT,
        img_w=Config.IMG_WIDTH,
        patch_size=Config.PATCH_SIZE,
        embed_dim=Config.EMBED_DIM,
        num_classes=Config.NUM_CLASSES,
        depth=Config.TRANSFORMER_LAYERS,
        num_heads=Config.TRANSFORMER_HEADS,
        mlp_ratio=4.0,
        attn_p=Config.TRANSFORMER_DROPOUT,
        mlp_p=Config.TRANSFORMER_DROPOUT,
        proj_p=Config.TRANSFORMER_DROPOUT
    )

    print_parameter_breakdown(model)
