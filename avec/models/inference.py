import torch
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
from avec.data.dataset import OpenFaceDatasetLSTM, collate_fn
import pandas as pd
from torch.utils.data import DataLoader
from avec import settings
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from avec.models.training import Trainer
import torch.optim as optim
from torch.optim import lr_scheduler
from avec.models import LSTM
from avec.models.training import train_LSTM

class Predictor():
    """
    Predictor for binary classification problems. 
    """

    def __init__(self, model):
        """ 
        model   = the model class to be instantiated, not the instantiated 
                  class itself
        """
        self.model = model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if not next(model.parameters()).is_cuda:
            self.model = self.model.to(self.device)

    def _list_from_tensor(self, tensor):
        if tensor.numel() == 1:
            return [tensor.item()]
        return list(tensor.cpu().detach().numpy())
    
    def predict(self, dataloader, threshold=0.5):
        self.model.eval()
        logit_threshold = torch.tensor(threshold / (1 - threshold)).log()
        logit_threshold = logit_threshold.to(self.device)
        pred_labels = []
        test_labels = []

        for inputs, labels, lengths in dataloader:
            inputs = inputs.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs, lengths)
            preds = outputs > logit_threshold
            pred_labels.extend(self._list_from_tensor(preds))
            test_labels.extend(self._list_from_tensor(labels))

        print("PREDS: ", pred_labels)
        print("LABELS: ", test_labels)
        # metrics = precision_recall_fscore_support(
        #     y_true=test_labels,
        #     y_pred=pred_labels,
        #     average="binary")
        # return np.array(metrics[:-1])
        report = classification_report(y_true=test_labels, y_pred=pred_labels)
        print(report)

def print_metrics(metrics):
    print("----------------------")        
    print("= For Class 1 =======")
    print("\t Precision: {} \t Recall: {} \t F1: {}".format(
        metrics[0], metrics[1], metrics[2]))

def predict_LSTM(feature, threshold=0.5):
    model = train_LSTM(feature)
    val_dataset = OpenFaceDatasetLSTM("val", feature)
    val_loader = DataLoader(val_dataset,
                            batch_size=8,
                            collate_fn=collate_fn)

    predictor = Predictor(model)
    metrics = predictor.predict(val_loader, threshold)
    #print_metrics(metrics)