from transformers import AutoImageProcessor, ViTModel

import torch
import torch.nn as nn
# from torchvision.models.vision_transformer import VisionTransformer
from baseline.vision_transformer import VisionTransformer
from einops import rearrange

class TemporalVisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=10, seq_length=61, embed_dim=768, num_heads=5, num_layers=6, num_classes=1):
        super().__init__()
        
        # Use the VisionTransformer from torchvision
        self.vit = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=in_channels,
            mlp_dim=128,
            num_classes=num_classes
        )
        
        # Adjust for the temporal data
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        
        # Positional encoding for temporal dimension
        # self.temporal_encoding = nn.Parameter(torch.randn(seq_length, embed_dim))
        
        # Decoder to produce segmentation maps
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2, 1, kernel_size=1)
        )
        
    def forward(self, x):
        print(x.size())
        # Input shape: [batch, seq_length, channels, height, width]
        # batch_size = x.size(0)
        
        # Flatten patches
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        print(x.size())

        # # Add temporal positional encoding
        # temporal_pos = self.temporal_encoding.unsqueeze(1).expand(-1, x.size(1), -1)
        # x = x + temporal_pos.repeat(batch_size, 1, 1)
        # print(x.size())

        # Pass through VisionTransformer (flattened along batch axis)
        x = self.vit(x)  # Pass through ViT's encoder
        print("after vit : ", x.size())
        x = rearrange(x, '(b t) -> b t c')
        # Reshape and decode back into the segmentation map
        # x = rearrange(x, '(b t) -> b e t p', b=batch_size)
        # x = self.decoder(x.mean(dim=2).view(batch_size, self.embed_dim, self.seq_length, self.seq_length))
        
        return x
