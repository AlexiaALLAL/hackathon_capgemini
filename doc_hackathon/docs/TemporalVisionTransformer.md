# 3. Implementation of a temporal ViT from scratch

You can find this implementation in `baseline\TemporalVisionTransformer.py`.
We tried to implement the time dependency starting from `torchvision`'s ViT model, but realized this wasn't going to be a successful approach given the amount of time we had.


The file contains a `TemporalVisionTransformer` class, which adapts a Vision Transformer for processing temporal sequences of images. Here’s a structured documentation for this class and its components:

---

### Module: `TemporalVisionTransformer`

#### Overview
`TemporalVisionTransformer` is a PyTorch model designed to handle temporal image sequences. This model leverages a Vision Transformer (ViT) as the backbone for feature extraction, with added support for temporal sequence processing, allowing it to capture patterns across time as well as spatial features.

#### Dependencies
```python
from transformers import AutoImageProcessor, ViTModel
import torch
import torch.nn as nn
from baseline.vision_transformer import VisionTransformer
from einops import rearrange
```

#### Class: `TemporalVisionTransformer`

```python
class TemporalVisionTransformer(nn.Module):
```

##### Description
The `TemporalVisionTransformer` class is a neural network model for temporal image sequence analysis, inheriting from `nn.Module`. It integrates a Vision Transformer (ViT) adapted for temporal data processing, with positional encoding for the temporal dimension to encode time-related information.

##### Parameters
- `img_size` (int): Size of the input image (assumed square). Default is 128.
- `patch_size` (int): Size of patches within the image. Default is 16.
- `in_channels` (int): Number of input channels per image in the sequence. Default is 10.
- `seq_length` (int): Number of images in the sequence. Default is 61.
- `embed_dim` (int): Embedding dimension for transformer. Default is 768.
- `num_heads` (int): Number of attention heads in each transformer layer. Default is 5.
- `num_layers` (int): Number of layers in the transformer. Default is 6.
- `num_classes` (int): Number of output classes. Default is 1.

##### Attributes
- `vit` (`VisionTransformer`): The core Vision Transformer module for extracting features from each image.
    - Configured with `img_size`, `patch_size`, `num_layers`, `num_heads`, `in_channels`, `embed_dim`, and `num_classes`.
- `seq_length` (int): Length of the temporal sequence.
- `embed_dim` (int): Dimension for embedding in the temporal context.
- `temporal_positional_encoding` (`nn.Parameter`): Positional encoding for each frame in the sequence, enhancing temporal awareness in the transformer model.
- `decoder` (`nn.Sequential`): Decoder to upsample the features and output a segmentation map

##### Methods
- `forward(x: torch.Tensor) -> torch.Tensor`
    - Forward pass for the temporal Vision Transformer.
    - **Parameters:**
        - `x` (`torch.Tensor`): Input tensor of shape `(batch_size, seq_length, in_channels, img_size, img_size)`.
    - **Returns:**
        - `torch.Tensor`: Output tensor for classification or regression tasks, typically of shape `(batch_size, num_classes)`.

##### Example Usage
```python
model = TemporalVisionTransformer(img_size=128, patch_size=16, in_channels=10, seq_length=61, embed_dim=768)
input_tensor = torch.randn(1, 61, 10, 128, 128)  # Example input
output = model(input_tensor)
print(output.shape)  # Expected output shape depends on num_classes (default is (1, 1))
```

##### Notes
- This model includes a ViT backbone for spatial feature extraction, with adjustments for handling sequences in the temporal dimension.
- Temporal positional encoding would allows the model to capture dependencies across the sequence length, which may improve performance on tasks involving sequential data.
