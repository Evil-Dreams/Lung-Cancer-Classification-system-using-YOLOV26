"""
UNETR (UNet Transformer) implementation for 3D medical image segmentation.
Based on: https://arxiv.org/abs/2103.10504
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, Union
import math

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock, UnetBasicBlock, UnetResBlock, get_output_padding, get_padding
from monai.networks.layers import Conv, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import

from src.utils.config import cfg

class UNETR(nn.Module):
    """
    UNETR (UNet Transformer) for 3D medical image segmentation.
    
    Args:
        in_channels: Dimension of input channels.
        out_channels: Dimension of output channels (number of classes).
        img_size: Dimension of input image (D, H, W).
        feature_size: Dimension of network feature size.
        hidden_size: Dimension of hidden layer in the transformer.
        mlp_dim: Dimension of feedforward layer in the transformer.
        num_heads: Number of attention heads.
        pos_embed: Type of position embedding ("conv" or "perceptron").
        norm_name: Feature normalization type and arguments.
        conv_block: Whether to use convolutional blocks in the decoder.
        res_block: Whether to use residual blocks in the decoder.
        dropout_rate: Dropout rate.
        spatial_dims: Number of spatial dimensions (2 or 3).
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        img_size: Union[Sequence[int], int] = (256, 256),  # Default to 2D
        feature_size: int = 16,
        hidden_size: int = 256,  # Reduced for 2D
        mlp_dim: int = 512,      # Reduced for 2D
        num_heads: int = 8,      # Reduced for 2D
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
            
        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        
        # Encoder part
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size * 2,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size * 4,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 8,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # Decoder part
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # Output layers
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        
        # Projection layers for skip connections
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def proj_feat(self, x):
        """Project features from transformer to spatial dimensions."""
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        new_axes = [0, len(x.shape) - 1] + [d + 1 for d in range(len(self.feat_size))]
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, x_in):
        # Get input shape
        x, hidden_states_out = self.vit(x_in)
        
        # Get encoder features
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Get decoder features
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        out = self.decoder1(out)
        
        # Get final output
        logits = self.out(out)
        
        return logits


class ViT(nn.Module):
    """
    Vision Transformer (ViT) for feature extraction in UNETR.
    """
    
    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        pos_embed: str,
        classification: bool = False,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
            
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        
        if self.classification:
            self.class_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid(),
            )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        hidden_states_out = []
        
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
            
        x = self.norm(x)
        
        if self.classification:
            x = self.class_head(x[:, 0])
            
        return x, hidden_states_out


class PatchEmbeddingBlock(nn.Module):
    """
    Patch embedding block for Vision Transformer.
    """
    
    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int,
        num_heads: int,
        pos_embed: str,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
            
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        
        self.spatial_dims = spatial_dims
        self.hidden_size = hidden_size
        
        # Image and patch size
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        
        # Calculate number of patches
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
                
        self.n_patches = 1
        for i in range(spatial_dims):
            self.n_patches *= img_size[i] // patch_size[i]
        
        # Patch embedding
        self.patch_embeddings = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Position embedding
        if pos_embed == "conv":
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, self.n_patches, hidden_size)
            )
        elif pos_embed == "perceptron":
            # Use a small MLP to generate position embeddings
            self.position_embeddings = nn.Sequential(
                nn.Linear(spatial_dims, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size),
            )
            
            # Generate position indices
            pos_indices = []
            for i in range(spatial_dims):
                pos = torch.arange(img_size[i] // patch_size[i])
                pos = pos.unsqueeze(0).repeat(self.n_patches // (img_size[i] // patch_size[i]), 1)
                pos = pos.transpose(0, 1).contiguous().view(-1)
                pos_indices.append(pos.float())
                
            pos_indices = torch.stack(pos_indices, dim=1)
            self.register_buffer("pos_indices", pos_indices)
        else:
            raise ValueError(f"Position embedding type {pos_embed} not recognized.")
        
        self.pos_embed = pos_embed
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get patch embeddings
        x = self.patch_embeddings(x)
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(-1, -2)
        
        # Add position embeddings
        if self.pos_embed == "conv":
            x = x + self.position_embeddings
        else:
            pos_embed = self.position_embeddings(self.pos_indices.to(x.device))
            x = x + pos_embed.unsqueeze(0)
        
        # Add class token if needed
        if hasattr(self, 'class_token'):
            class_token = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.cat((class_token, x), dim=1)
        
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and feed-forward network.
    """
    
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        # Self-attention
        h = x
        x = self.attention_norm(x)
        x, _ = self.attention(x, x, x)
        x = x + h
        
        # Feed-forward network
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        
        return x


def get_unetr_model() -> UNETR:
    """
    Create a UNETR model with default configuration.
    
    Returns:
        UNETR: UNETR model instance
    """
    spatial_dims = 2  # Force 2D for compatibility with the trained model
    img_size = [256, 256]  # Fixed size to match the trained model
    
    return UNETR(
        in_channels=1,  # Grayscale input
        out_channels=2,  # Background + nodule
        img_size=img_size,
        feature_size=16,
        hidden_size=256,  # Matches the trained model
        mlp_dim=512,      # Matches the trained model
        num_heads=8,      # Matches the trained model
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.1,
        spatial_dims=spatial_dims,
    )


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a test input
    batch_size = 2
    in_channels = 1
    img_size = (96, 96, 96)
    x = torch.randn(batch_size, in_channels, *img_size).to(device)
    
    # Create model
    model = UNETR(
        in_channels=in_channels,
        out_channels=2,
        img_size=img_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
        spatial_dims=3,
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
