import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18

nclasses = 43  # GTSRB as 43 classes


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = resnet18()
        self.resnet.fc = nn.Linear(512, nclasses)

    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x)
