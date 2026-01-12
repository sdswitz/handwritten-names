import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbed(nn.Module):
    """
    Image to Patch Embeddings via Convolution
    """
    def __init__(self, img_h, img_w, patch_size=16, in_chans=1, embed_dim=256, bias=True):
        
        """
        Args:
            img_size: Expected Image Shape (img_size x img_size)
            patch_size: Wanted size for each patch
            in_chans: Number of channels in image (1 for grayscale)
            embed_dim: Transformer embedding dimension
        
        """
        super().__init__()
        assert img_h % patch_size == 0 and img_w % patch_size == 0
        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_h // patch_size) * (img_w // patch_size)
        
        self.proj = nn.Conv2d(in_channels=in_chans,
                              out_channels=embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size,
                              bias=bias)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x


class SelfAttentionEncoder(nn.Module):
    """
        Self Attention Proposed in `Attention is All  You Need` - https://arxiv.org/abs/1706.03762
    """

    def __init__(self,
                embed_dim=256,
                num_heads=12, 
                attn_p=0.0,
                proj_p=0.0,
                fused_attn=False):
        """
        
        Args:
            embed_dim: Transformer Embedding Dimension
            num_heads: Number of heads of computation for Attention 
            attn_p: Probability for Dropout2d on Attention cube
            proj_p: Probability for Dropout on final Projection
        """

        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn  

        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.attn_p = attn_p
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)

        if self.fused_attn:
          x = F.scaled_dot_product_attention(q,k,v, dropout_p=self.attn_p)
        else:
            attn = (q @ k.transpose(-2,-1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        
        x = x.transpose(1,2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
  

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, mlp_p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(mlp_p)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(mlp_p)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, proj_p=0.0, attn_p=0.0, mlp_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttentionEncoder(embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        attn_p=attn_p,
                                        proj_p=proj_p)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim,
                    hidden_features=hidden_features,
                    out_features=embed_dim,
                    mlp_p=mlp_p)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self,
                 img_h=128,
                 img_w=512,
                 patch_size=16,
                 in_chans=1,
                 num_classes=37,
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=2.0,
                 attn_p=0.0,
                 mlp_p=0.0,
                 proj_p=0.0,
                 pos_p=0.0,
                 head_p=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_h=img_h,
                                      img_w=img_w,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim)
            
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(pos_p)
        
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim=embed_dim,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         proj_p=proj_p,
                         attn_p=attn_p,
                         mlp_p=mlp_p)
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(head_p)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def _init_weights(self, module: nn.Module):

        if isinstance(module, ViT):
            module.pos_embed.data = nn.init.trunc_normal_(module.pos_embed.data, mean=0, std=0.02)

        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        # x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # x = self.head_drop(x)
        # x = self.head(x)
        output = self.head(x)
        output = output.permute(1,0,2)
        output = torch.nn.functional.log_softmax(output, dim=2)
        
        batch_size = x.size(0)
        output_lengths = torch.full(
            (batch_size,),
            fill_value=output.size(0),
            dtype=torch.long
        )
        
        return output, output_lengths