import torch
import torch.nn as nn
from torchvision.models import resnet50

class ConvNetwork(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ConvNetwork, self).__init__()

        self.base_model = resnet50(pretrained=False)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.base_model.fc = nn.Sequential(
            nn.Linear(1000, 1)
        )
        # self.linear = torch.nn.Linear(1000, 1)
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """   
        #y_pred = nn.Sigmoid(self.linear(self.base_model.fc))
        x = self.base_model(x)
        # y_pred = self.linear(self.base_model.fc)
        return x

model = ConvNetwork()