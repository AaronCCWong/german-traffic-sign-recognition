import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43  # GTSRB as 43 classes


class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv2_drop = nn.Dropout2d()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv_bn2 = nn.BatchNorm2d(256)
        self.conv4_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(6400, 512)
        self.fc1_bn = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, nclasses)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(
            self.conv_bn1(self.conv2(x))), 2))

        x = self.conv3(x)
        x = F.relu(F.max_pool2d(self.conv4_drop(
            self.conv_bn2(self.conv4(x))), 2))

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training)

        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.dropout(x, training=self.training)

        x = self.fc3(x)
        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
