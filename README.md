# Mines 2024 Data Challenge - Team 6 Repo

Welcome to the **Team 6 Data Challenge Repo**! This repository shows our researchs during the week. For the trainings we mainly used google collab and a drive (https://drive.google.com/drive/folders/1TftEbvydR-n1Ca8ADERiAlYj7tO3Jq42?usp=drive_link)


## Experiments

### 1. yolo implementation
    You can find it in the yolo folder. We tried to run yolo on our data but quickly stop due to yolo being trained with polygonal bounding box for segmentation. Since our data is only annotated with mask of the classes on our images, it is not possible to directly feed the annotations to yolo. 
    While trying to implement this code, we chose to give up on the temporal dimension selection only one image per sequence and choose only the RBG channel in order to feed the images to yolo. 

### 2. Implementation of a ViT pretrained on a crop segmentation task
    You can find this implementation in prithvi_notebook.ipynb. Here we used a promising network already pretrained and adaptated to our task, yet we weren't able to retreive all the code, so we had to recode a part of the network. It resulted in a working implementation that takes too long to train and gives poor results

### 3. Implementation of a temporal ViT from scratch 
    You can find this implementation in baseline\model_vision_transformer.py. Here we tried to implement the time dependency starting from torchvision ViT's model, but realized this wasn't going to success in the amount of time we had

### 4. Implementation of a simple vision transformer whithout time dependency from scratch 
    You can find this implementation in baseline\vision_transformer.py and in the drive. Here we tried to implement the simplest ViT we could using only one image by sequence. We had to modify the classification head of the network to fulfill the task of segmentation. This model was trained on our data and resulted in a 8% mIoU on the visible part of the test set. It is our best performing implementation, yet we were able to get 10% mIoU by running only baseline/model.py on a few epochs, which shows the limit of this ViT network.

That's all for our research
