from transformers import AutoImageProcessor, ViTModel

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer
from einops import rearrange

class TemporalVisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=10, seq_length=61, embed_dim=768, num_heads=8, num_layers=6, num_classes=1):
        super().__init__()
        
        # Use the VisionTransformer from torchvision
        self.vit = VisionTransformer(
            image_size=img_size, 
            patch_size=patch_size, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            hidden_dim=embed_dim, 
            mlp_dim=embed_dim * 4, 
            num_classes=num_classes
        )
        
        # Adjust for the temporal data
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        
        # Custom patch embedding to handle temporal data
        self.patch_embedding = nn.Conv3d(in_channels, embed_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))
        
        # Positional encoding for temporal dimension
        self.temporal_encoding = nn.Parameter(torch.randn(seq_length, embed_dim))
        
        # Decoder to produce segmentation maps
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2, 1, kernel_size=1)
        )
        
    def forward(self, x):
        # Input shape: [batch, seq_length, channels, height, width]
        batch_size = x.size(0)
        
        # Reshape and apply patch embedding
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.patch_embedding(x)  # [batch, embed_dim, seq_length, num_patches, num_patches]
        
        # Flatten patches
        x = rearrange(x, 'b e t h w -> (b t) (h w) e')
        
        # Add temporal positional encoding
        temporal_pos = self.temporal_encoding.unsqueeze(1).expand(-1, x.size(1), -1)
        x = x + temporal_pos.repeat(batch_size, 1, 1)
        
        # Pass through VisionTransformer (flattened along batch axis)
        x = rearrange(x, '(b t) p e -> (b t) p e', b=batch_size)
        x = self.vit.encoder(x)  # Pass through ViT's encoder
        
        # Reshape and decode back into the segmentation map
        x = rearrange(x, '(b t) p e -> b e t p', b=batch_size)
        x = self.decoder(x.mean(dim=2).view(batch_size, self.embed_dim, self.seq_length, self.seq_length))
        
        return x
