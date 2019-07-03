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
import scipy.io

__all__ = ["OpenFaceDatasetLSTM"]

class OpenFaceDatasetLSTM(Dataset):
    "Class abstraction to load OpenFace features"
    def __init__(self, subset="train", dataset="OpenFace"):

        if subset == "val":
            path = settings.PATH_TO_VALIDATION_DATA
        elif subset == "train":
            path = settings.PATH_TO_TRAINING_DATA
        elif subset == "test":
            path = settings.PATH_TO_TEST_DATA
        else:
            raise ValueError("No such subset {} file exist".format(subset))

        open_face_path = os.path.join(path, "*{}*".format(dataset))
        open_faces_files = glob.glob(open_face_path)

        self.max_size = 0
        self.metadata = []

        # min_max_scaler = self.load_min_max_scaler()
        # if min_max_scaler:
        #     min_max_scaler = self.load_min_max_scaler()
        # else:
        #     min_max_scaler = self.calculate_min_max_scaler(open_faces_files)
        #     self.save_min_max_scaler(min_max_scaler)
        

        self.features_list = []
        self.label_list = []

        labels_path = os.path.join(path, "labels.csv")
        label_dataframe = pd.read_csv(labels_path, encoding="utf-8")

        for f in open_faces_files:
            #np_array = pd.read_csv(f, encoding="utf-8").values
            np_array = self.load_data(f)
            participant_id = "_".join(f.split("_")[:2]).split(os.sep)[-1]
            label_array = label_dataframe.loc[
                label_dataframe.Participant_ID == participant_id].PHQ_Binary.values
            self.label_list.append(label_array.item(0))
            # np_array = min_max_scaler.transform(np_array)
            torch_array = torch.from_numpy(np_array).float()
            self.features_list.append(torch_array)
            if torch_array.size(0) > self.max_size:
                self.max_size = torch_array.size(0)
        
        self.examples_sizes = list(map(lambda x: x.size(0), self.features_list))
        #self.examples_sizes = torch.tensor(self.examples_sizes).long()
        #self.features_list = list(map(self.pad_example, self.features_list))
        #self.features_list = torch.stack(self.features_list).float()
    
    def load_data(self, file):

        if "OpenFace" in file:
            np_array = pd.read_csv(file, encoding="utf-8").values
            np_array = np_array[:, 4:]
        elif "ResNet" in file:
            np_array = scipy.io.loadmat(file)["feature"]
        
        return np_array

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

    # def pad_example(self, video):
    #     i,j = video.size(0), video.size(1)
    #     pad_size = self.max_size - i
    #     padd = torch.zeros([pad_size, j], dtype=video.dtype)
    #     padded = torch.cat([video, padd])
    #     return padded

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        return (self.features_list[idx], self.label_list[idx],
                self.examples_sizes[idx])

def collate_fn(data):
    """This function will be used to pad the tweets to max length
       in the batch and transpose the batch from 
       batch_size x max_seq_len to max_seq_len x batch_size.
       It will return padded vectors, labels and lengths of each tweets (before padding)
       It will be used in the Dataloader
    """
    data.sort(key=lambda x: x[2], reverse=True)
    _, labels, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = data[0][0].size(1)
    features = torch.zeros((len(data), max_len, n_ftrs))
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

    return features.float(), labels.long(), lengths.long()


#ds = OpenFaceDatasetLSTM("val", "OpenFace")
#dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
