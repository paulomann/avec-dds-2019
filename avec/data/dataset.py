from torch.utils.data import Dataset
from avec import settings
from torch.utils.data import DataLoader
import glob
import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

class OpenFaceDataset(Dataset):
    "Class abstraction to load OpenFace features"
    def __init__(self, subset="train"):

        if subset == "val":
            path = settings.PATH_TO_VALIDATION_DATA
        elif subset == "train":
            path = settings.PATH_TO_TRAINING_DATA
        elif subset == "test":
            path = settings.PATH_TO_TEST_DATA
        else:
            raise ValueError("No such subset {} file exist".format(subset))

        path = os.path.join(path, "*OpenFace*")
        open_faces_files = glob.glob(path)

        self.size = 0
        self.metadata = []

        min_max_scaler = self.load_min_max_scaler()
        if min_max_scaler:
            min_max_scaler = self.load_min_max_scaler()
        else:
            min_max_scaler = self.calculate_min_max_scaler(open_faces_files)
            self.save_min_max_scaler(min_max_scaler)

        self.features_list = []
        for f in open_faces_files:
            np_array = pd.read_csv(f, encoding="utf-8").values
            self.metadata.append(torch.from_numpy(np_array[:, 0:4]))
            np_array = np_array[:, 4:]
            np_array = min_max_scaler.transform(np_array)
            torch_array = torch.from_numpy(np_array)
            self.features_list.append(torch_array)
            if torch_array.size(0) > self.size:
                self.size = torch_array.size(0)
    

    def load_min_max_scaler(self):
        try:
            with open(settings.PATH_TO_MIN_MAX_SCALER, "rb") as fp:
                return pickle.load(fp)
        except:
            return None

    def calculate_min_max_scaler(self, files_list):
        features_list = self.get_all_features_list(files_list)
        cat_tensors = np.concatenate(features_list, axis=0)
        scaler = MinMaxScaler()
        scaler.fit(cat_tensors)
        return scaler

    def get_all_features_list(self, files_list):
        features_list = []
        for f in files_list:
            np_array = pd.read_csv(f, encoding="utf-8").values
            features_list.append(np_array[:, 4:])
        return features_list

    def save_min_max_scaler(self, scaler):
        with open(settings.PATH_TO_MIN_MAX_SCALER, "wb") as fp:
            pickle.dump(scaler, fp)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.features_list[idx]

ds = OpenFaceDataset("train")