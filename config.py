import torch

DATA_PATH = "/Users/gatam/Documents/Mines/3A/IDSC/data_hackathon/DATA-mini"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
