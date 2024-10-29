# 4. Implementation of a simple vision transformer whithout time dependency from scratch

You can find this implementation in `baseline\SegmentationViT.py` and in the Drive folder.
We tried to implement the simplest ViT we could using only one image by sequence. We had to modify the classification head of the network to fulfill the task of segmentation.
This implementation uses the VisionTransformer from torchvision.models. In order to atapt it for our dataset, we had to change the number of channels in the ViT... (in_channels = 10) that was hardcoded in the original implementation. You can find the modified version in `baseline\vision_transformer.py`.

This model was trained on our data and resulted in a 8% mIoU on the visible part of the test set. It is our best performing implementation, yet we were able to get 10% mIoU by running only `baseline/model.py` on a few epochs, which shows the limit of this ViT network.




The file implements a `SegmentationViT` class, which is a Vision Transformer (ViT) model adapted for image segmentation tasks. Here’s a structured documentation for this class and its components:

---

### Module: `SegmentationViT`

#### Overview
`SegmentationViT` is a PyTorch model designed for image segmentation tasks. This model leverages a Vision Transformer (ViT) as the backbone for feature extraction, specifically using only the 10th frame of a sequence as input. A lightweight decoder transforms the transformer outputs into a segmentation map.

#### Dependencies
```python
import torch
import torch.nn as nn
from baseline.vision_transformer import VisionTransformer
```

#### Class: `SegmentationViT`

```python
class SegmentationViT(nn.Module):
```

##### Description
The `SegmentationViT` class inherits from `nn.Module`. It applies a ViT-based architecture tailored for image segmentation, with a decoder for mapping encoded features to a segmented output. The model is configured for a specified input size, patch size, number of input channels, and number of segmentation classes.

##### Parameters
- `img_size` (int): Dimension of the input image, assumed square. Default is 128.
- `patch_size` (int): Size of each patch divided from the image. Default is 16.
- `in_channels` (int): Number of input channels in each image (e.g., 10 for a sequence). Default is 10.
- `n_classes` (int): Number of segmentation classes. Default is 20.

##### Attributes
- `vit` (`VisionTransformer`): The backbone model for feature extraction. The `VisionTransformer` component is imported from an external module, configured with:
    - `image_size`: Input image size (`img_size`).
    - `patch_size`: Size of patches (`patch_size`).
    - `in_channels`: Number of input channels.
    - `num_layers`, `num_heads`, `hidden_dim`, and `mlp_dim`: Parameters defining transformer depth, attention heads, hidden dimensions, and feedforward layer size.
- `decoder` (`nn.Sequential`): Decoder to upsample the features and output a segmentation map. It consists of:
    - `ConvTranspose2d`: Deconvolution layer for upsampling.
    - `ReLU`: Activation function.
    - Additional convolutional and upsampling layers (depending on final dimensions) to map ViT output to segmentation labels.

##### Methods
- `forward(x: torch.Tensor) -> torch.Tensor`
    - Forward pass for the segmentation model.
    - **Parameters:**
        - `x` (`torch.Tensor`): Input tensor of shape `(batch_size, in_channels, img_size, img_size)`.
    - **Returns:**
        - `torch.Tensor`: Segmentation map output of shape `(batch_size, n_classes, img_size, img_size)`.

##### Example Usage
```python
model = SegmentationViT(img_size=128, patch_size=16, in_channels=10, n_classes=20)
input_tensor = torch.randn(1, 10, 128, 128)
output = model(input_tensor)
print(output.shape)  # Expected output shape: (1, 20, 128, 128)
```

##### Notes
- This model is specifically structured to use only the 10th frame from a sequence, meaning it may need adaptation for real-time or frame-dependent applications.
- The ViT is used as a feature extractor, while the decoder is designed to handle the transformation of features into a spatial segmentation map.