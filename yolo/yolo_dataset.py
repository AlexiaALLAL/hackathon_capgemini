"""
Baseline Pytorch Dataset
"""

import os
from pathlib import Path
import matplotlib.image
import matplotlib.pyplot as plt

import geopandas as gpd
import numpy as np
import torch


class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, folder: Path):
        super(BaselineDataset, self).__init__()
        self.folder = folder

        # Get metadata
        print("Reading patch metadata ...")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        print("Done.")

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index
        print("Dataset ready.")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        id_patch = self.id_patches[item]

        # Open and prepare satellite data into T x C x H x W arrays
        path_patch = os.path.join(self.folder, "DATA_S2", "S2_{}.npy".format(id_patch))
        data = np.load(path_patch).astype(np.float32)
        data = {"S2": torch.from_numpy(data)}

        # If you have other modalities, add them as fields of the `data` dict ...
        # data["radar"] = ...

        # Open and prepare targets
        target = np.load(
            os.path.join(self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch))
        )
        target = torch.from_numpy(target[0].astype(int))

        return data, target


class BaselineDatasetTest(torch.utils.data.Dataset):
    def __init__(self, folder: Path):
        super(BaselineDatasetTest, self).__init__()
        self.folder = folder

        # Get metadata
        print("Reading patch metadata ...")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        print("Done.")

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index
        print("Dataset ready.")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        id_patch = self.id_patches[item]

        # Open and prepare satellite data into T x C x H x W arrays
        path_patch = os.path.join(self.folder, "DATA_S2", "S2_{}.npy".format(id_patch))
        data = np.load(path_patch).astype(np.float32)
        data = {"S2": torch.from_numpy(data)}

        # If you have other modalities, add them as fields of the `data` dict ...
        # data["radar"] = ...

        return data

def get_rgb(im) -> np.ndarray:
    """
    Retrieves an RGB image from a Sentinel-2 time series.

    This function extracts the RGB (Red, Green, Blue) channels from a Sentinel-2
    time series, normalizes the values between 0 and 1, and prepares the image
    for display by rearranging the axes to match the standard image format.

    Args:
        x (torch.Tensor): The input time series data with shape
            (batch_size, time_steps, channels, height, width). The expected
            channels are ordered as [Blue, Green, Red, ...].
        batch_index (int, optional): Index of the batch to extract the image
            from. Defaults to 0.
        t_show (int, optional): The time step index to show. Defaults to 0.

    Returns:
        np.ndarray: A normalized RGB image as a NumPy array with shape
        (height, width, 3), ready for visualization.

    Example:
        >>> rgb_image = get_rgb(x, batch_index=0, t_show=2)
        >>> plt.imshow(rgb_image)
        >>> plt.show()

    Notes:
        - The RGB image is created from the Red, Green, and Blue bands (channels 2, 1, 0).
        - The pixel values are normalized to the range [0, 1].
        - The channel axes are rearranged to make the image displayable by matplotlib.
    """

    
    mx = im.max(axis=(1, 2))
    mi = im.min(axis=(1, 2))
    im = (im - mi[:, None, None]) / (mx - mi)[:, None, None]
    im = im.swapaxes(0, 2).swapaxes(0, 1)
    im = np.clip(im, a_max=1, a_min=0)
    return im




for id_patch in range(10000,10010):
    path_patch = os.path.join( "D:/data-challenge-invent-mines-2024/DATA-mini/DATA-mini/ANNOTATIONS", "TARGET_{}.npy".format(id_patch))
    data = np.load(path_patch).astype(np.float32)
    print(data.shape)
    print(data[0])
    #plt.imshow(data[0])
    #plt.show()
    data = data[0]
    #data = get_rgb(data[10,[2,1,0]])
    #print(data.shape)
    save_path = os.path.join( "D:/data-challenge-invent-mines-2024/DATA-mini/DATA-mini/data_yolo/labels", "S2_{}.txt".format(id_patch))
    #print(type(save_path))
    np.savetxt(save_path, data)

