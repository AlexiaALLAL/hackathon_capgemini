import torch
import torch.nn as nn
# from torchvision.models.vision_transformer import VisionTransformer
from baseline.vision_transformer import VisionTransformer

class SegmentationViT(nn.Module):
    """
    More simple model : takes only the 10th image of the sequence and predicts the segmentation map.
    """
    def __init__(self, img_size=128, patch_size=16, in_channels=10, n_classes=20):
        super().__init__()
        
        # Use the VisionTransformer from torchvision
        self.vit = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=6,
            num_heads=4,
            hidden_dim=768,
            mlp_dim=128
            # num_classes=1000
        )
        
        # Decoder to convert the transformer output to a segmentation map
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, n_classes, kernel_size=1),
            nn.Upsample(scale_factor=img_size//(patch_size*2), mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # print(x.shape)
        # Forward through the Vision Transformer
        B, C, H, W = x.shape  # [batch, seq_length, channels, height, width]
        x = self.vit(x)  # (B, N, D) where N is the number of patches and D is embedding dim
        # print("output after vit : ", x.shape)

        
        # Reshape and decode to get segmentation output
        grid_size = H // self.vit.patch_size
        x = x.view(B, grid_size, grid_size, -1).permute(0, 3, 1, 2)
        # grid_size = H // 16
        # x = x.view(B, -1, H, W)
        # x = x.permute(0, 2, 1).view(B, -1, H // 16, W // 16)
        # print("output size after permute and view :", x.shape)
        x = self.decoder(x)
        # print("output size after decoder :", x.shape)
        return x
