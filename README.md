# Mines 2024 Data Challenge - Team 6 Repo

Welcome to the **Team 6 Data Challenge Repo**! This repository shows our researchs during the week. For the trainings we mainly used google Colab and a drive (https://drive.google.com/drive/folders/1TftEbvydR-n1Ca8ADERiAlYj7tO3Jq42?usp=drive_link)

## Experiments

### 1. YOLO implementation

You can find it in the `yolo/` folder.
We tried to run YOLO on our data but quickly stopped due to YOLO being trained with polygonal bounding boxes for segmentation. Since our data is only annotated with the masks of the classes on our images, it is not possible to directly feed these annotations to YOLO.

While trying to implement this code, we chose to give up on the temporal dimension, selecting only one image per sequence and choosing only the RBG channels in order to feed the images to YOLO.

### 2. Implementation of a ViT pretrained on a crop segmentation task

You can find this implementation in `prithvi_notebook.ipynb`.
Here, we used a promising network already pretrained and adaptated to our task, yet we weren't able to retrieve all the code, so we had to recode part of the network. It resulted in a working implementation that takes too long to train and gives poor results on the few epochs on which it was trained.

### 3. Implementation of a temporal ViT from scratch

You can find this implementation in `baseline\TemporalVisionTransformer.py`.
We tried to implement the time dependency starting from `torchvision`'s ViT model, but realized this wasn't going to be a successful approach given the amount of time we had.

### 4. Implementation of a simple vision transformer whithout time dependency from scratch

You can find this implementation in `baseline\SegmentationViT.py` and in the Drive folder.
We tried to implement the simplest ViT we could using only one image by sequence. We had to modify the classification head of the network to fulfill the task of segmentation.
This implementation uses the VisionTransformer from torchvision.models. In order to atapt it for our dataset, we had to change the number of channels in the ViT... (in_channels = 10) that was hardcoded in the original implementation. You can find the modified version in `baseline\vision_transformer.py`.

This model was trained on our data and resulted in a 8% mIoU on the visible part of the test set. It is our best performing implementation, yet we were able to get 10% mIoU by running only `baseline/model.py` on a few epochs, which shows the limit of this ViT network.
