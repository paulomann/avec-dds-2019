from torch.utils.data import Dataset
from avec import settings
from torch.utils.data import DataLoader
import glob
import os
import pandas as pd

class OpenFaceDataset(Dataset):
    "Class abstraction to load OpenFace features"
    def __init__(self, subset="train"):

        if subset == "val":
            path = settings.PATH_TO_DEVELOPMENT_DATA
        elif subset == "train":
            path = settings.PATH_TO_TRAINING_DATA
        elif subset == "test":
            path = settings.PATH_TO_TEST_DATA
        else:
            raise ValueError("No such subset {} file exist".format(subset))

        path = os.path.join(path, "*OpenFace*")
        open_faces_files = glob.glob(path)
        print(open_faces_files)
        self.dataframes = [pd.read_csv(f) for f in open_faces_files]

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass