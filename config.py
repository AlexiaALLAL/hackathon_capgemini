import torch

DATA_PATH_TRAIN = "/Users/alexi/Documents/mines/S5/idsc/capgemini challenge/DATA/DATA/TRAIN"
DATA_PATH_TEST = "/Users/alexi/Documents/mines/S5/idsc/capgemini challenge/DATA/DATA/TEST"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
