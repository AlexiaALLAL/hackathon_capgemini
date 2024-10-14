from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from baseline.collate import pad_collate
from baseline.dataset import BaselineDataset
from baseline.model import SimpleSegmentationModel
from baseline.submission_tools import masks_to_str
from baseline.train import print_iou_per_class, print_mean_iou
from config import DATA_PATH


def test_model(
        input_channels: int,
        nb_classes: int,
        data_folder: Path,
        batch_size: int = 1,
):
    # Load model
    model = SimpleSegmentationModel(input_channels, nb_classes)
    model = torch.load("checkpoints/model.pth")
    model.eval()

    # Load dataset
    dataset = BaselineDataset(data_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=pad_collate, shuffle=False
    )

    # Evaluate model
    all_targets = torch.zeros(len(dataloader), 128, 128)
    all_preds = torch.zeros(len(dataloader), 128, 128)
    for i, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            preds = model(images["S2"][:, 10, :, :, :])  # Only the 10th image
            preds = torch.argmax(preds, dim=1)
        
        all_preds[batch_size*i:batch_size*(i+1)] = preds
        all_targets[batch_size*i:batch_size*(i+1)] = targets

    all_targets_flat = all_targets.cpu().numpy().flatten()
    all_preds_flat = all_preds.cpu().numpy().flatten()
    
    # Print mIoU for the test set
    print_iou_per_class(all_targets_flat, all_preds_flat, nb_classes)
    print_mean_iou(all_targets_flat, all_preds_flat)

    # Generate the csv submission file
    masks = masks_to_str(all_preds)
    submission = pd.DataFrame.from_dict({"ID": range(len(all_preds)), "MASKS": masks})
    submission["ID"] = submission["ID"] + 20000
    submission.to_csv("submissions/submission1.csv", index=False)


if __name__ == "__main__":
    test_model(input_channels=10,
               nb_classes=20,
               data_folder=Path(DATA_PATH),
                batch_size=1,
               )
