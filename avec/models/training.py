import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import copy
import time
import json
from avec.data.dataset import OpenFaceDatasetLSTM, collate_fn
from avec.models import LSTM


# model = TODO

# criterion = nn.BCEWithLogitsLoss()

# parameters_for_training = filter(lambda p: p.requires_grad, model.parameters())

# optimizer_ft = optim.SGD(parameters_for_training, lr=0.01,
#                         momentum=0.9, weight_decay=0.0001, nesterov=True)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.85)

# train_dataset = OpenFaceDataset("train")
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# val_dataset = OpenFaceDataset("val")
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# dataloaders = {"train": train_loader, "val": val_loader}
# dataset_sizes = {"train": len(train_dataset), "val": len(train_dataset)}

# DEPOIS DE TREINAR - SALVAR
# torch.save(model.state_dict(), PATH)

class Trainer():

    def __init__(self, model, dataloaders, dataset_sizes, criterion, optimizer, 
                 scheduler, num_epochs=100, threshold=0.5):
        self.acc_loss = {"train": {"loss": [], "acc": []}, 
                         "val": {"loss": [], "acc": []}}
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        print("Using device ", self.device)
        # if torch.cuda.device_count() > 1:
        #     print("Using {} GPUs!".format(torch.cuda.device_count()))
        #     self.model = nn.DataParallel(model, device_ids=(0,3))
        self.dataset_sizes = dataset_sizes
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.logit_threshold = torch.tensor(threshold / (1 - threshold)).log()
        self.logit_threshold = self.logit_threshold.to(self.device)

    def train_model(self):

        since = time.time()

        self.acc_loss = {"train": {"loss": [], "acc": []}, 
                         "val": {"loss": [], "acc": []}}

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels, lengths in self.dataloaders[phase]:
                    lengths = lengths.to(self.device)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs, lengths)
                        #print("OUTPUTS: ", outputs)
                        # _, preds = torch.max(outputs, 1)
                        preds =  outputs > self.logit_threshold
                        print("PREDS: ", preds)
                        print("LABELS: ", labels)
                        loss = self.criterion(outputs, labels.float())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds.long() == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = (running_corrects.double() / 
                             self.dataset_sizes[phase])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                self.acc_loss[phase]["loss"].append(epoch_loss)
                self.acc_loss[phase]["acc"].append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model


def train_LSTM(feature):
    
    if feature == "OpenFace":
        input_size = 49
    elif feature == "ResNet":
        input_size = 2048
    else:
        raise ValueError

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lstm = LSTM(input_size, 64)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([3.],
        device=device))

    parameters_for_training = filter(
        lambda p: p.requires_grad, lstm.parameters())

    optimizer_ft = optim.SGD(parameters_for_training,
                              lr=0.001,
                              momentum=0.9,
                              nesterov=True,
                              weight_decay=0.005)

    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.85)

    train_dataset = OpenFaceDatasetLSTM("train", feature)
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              collate_fn=collate_fn)
    val_dataset = OpenFaceDatasetLSTM("val", feature)
    val_loader = DataLoader(val_dataset,
                            batch_size=8,
                            collate_fn=collate_fn)

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    trainer = Trainer(lstm,
                      dataloaders,
                      dataset_sizes,
                      criterion,
                      optimizer_ft,
                      scheduler,
                      num_epochs=2)

    return trainer.train_model()