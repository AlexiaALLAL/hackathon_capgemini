from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import jaccard_score
from tqdm import tqdm

from baseline.collate import pad_collate
from baseline.dataset import BaselineDataset
from baseline.model import SimpleSegmentationModel
from torchvision.models.vision_transformer import VisionTransformer
from baseline.SegmentationViT import SegmentationViT
from baseline.TemporalVisionTransformer import TemporalVisionTransformer
from config import DATA_PATH_TRAIN, DEVICE


def print_iou_per_class(
    targets: torch.Tensor,
    preds: torch.Tensor,
    nb_classes: int,
) -> None:
    """
    Compute IoU between predictions and targets, for each class.

    Args:
        targets (torch.Tensor): Ground truth of shape (B, H, W).
        preds (torch.Tensor): Model predictions of shape (B, nb_classes, H, W).
        nb_classes (int): Number of classes in the segmentation task.
    """

    # Compute IoU for each class
    # Note: I use this for loop to iterate also on classes not in the demo batch

    iou_per_class = []
    for class_id in range(nb_classes):
        iou = jaccard_score(
            targets == class_id,
            preds == class_id,
            average="binary",
            zero_division=0,
        )
        iou_per_class.append(iou)

    for class_id, iou in enumerate(iou_per_class):
        print(
            "class {} - IoU: {:.4f} - targets: {} - preds: {}".format(
                class_id, iou, (targets == class_id).sum(), (preds == class_id).sum()
            )
        )


def print_mean_iou(targets: torch.Tensor, preds: torch.Tensor) -> None:
    """
    Compute mean IoU between predictions and targets.

    Args:
        targets (torch.Tensor): Ground truth of shape (B, H, W).
        preds (torch.Tensor): Model predictions of shape (B, nb_classes, H, W).
    """

    mean_iou = jaccard_score(targets, preds, average="macro")
    print(f"meanIOU (over existing classes in targets): {mean_iou:.4f}")


def train_model(
    data_folder: Path,
    nb_classes: int,
    input_channels: int,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
) -> SimpleSegmentationModel:
    """
    Training pipeline.
    """
    # Create data loader
    dataset = BaselineDataset(data_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=pad_collate, shuffle=True
    )

    # Initialize the model, loss function, and optimizer
    model = SimpleSegmentationModel(input_channels, nb_classes)
    # model = SegmentationViT()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move the model to the appropriate device (GPU if available)
    device = torch.device(device)
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Move data to device
            inputs["S2"] = inputs["S2"].to(device)  # Satellite data
            targets = targets.long()
            targets = targets.to(device)
            

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # outputs = model(inputs["S2"]) # use all images
            outputs = model(inputs["S2"][:, 10, :, :, :])  # only use the 10th image
            # Loss computation
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Get the predicted class per pixel (B, H, W)
            preds = torch.argmax(outputs, dim=1)

            # Move data from GPU/Metal to CPU
            targets = targets.cpu().numpy().flatten()
            preds = preds.cpu().numpy().flatten()

            if verbose:
                # Print IOU for debugging
                # print_iou_per_class(targets, preds, nb_classes) # that is a bit much
                print_mean_iou(targets, preds)

        # Print the loss for this epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete.")
    torch.save(model, f"checkpoints/simple_model_epoch{num_epochs}.pth")
    return model


if __name__ == "__main__":
    # Example usage:
    model = train_model(
        data_folder=Path(DATA_PATH_TRAIN),
        nb_classes=20,
        input_channels=10,
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        device=DEVICE,
        verbose=False,
    )
