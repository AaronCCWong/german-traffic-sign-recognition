from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter


writer = SummaryWriter()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                    help='SGD momentum (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--network', choices=['base', 'deepbase', 'resnet'], default='base',
                    help='model to use (default: base')
args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, base_data_transforms, data_transforms, validation_data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set


if args.network == 'base' or args.network == 'deepbase':
    transform = base_data_transforms
    val_transform = base_data_transforms
else:
    transform = data_transforms
    val_transform = validation_data_transforms

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images', transform=transform),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images', transform=val_transform),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

### Neural Network and Optimizer
from base_model import BaseNet
from deep_model import DeepNet
from resnet_model import ResNet

if args.network == 'base':
    model = BaseNet()
elif args.network == 'deepbase':
    model = DeepNet()
elif args.network == 'resnet':
    model = ResNet()

model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

def train(epoch):
    model.train()
    total_imgs = 0
    accurate_count = 0
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)

        _, predicted = torch.max(output.data, 1)
        total_imgs += target.size(0)
        accurate_count += (predicted == target).sum().item()

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    accuracy = float(accurate_count) / float(total_imgs)
    writer.add_scalar('Avg Loss', sum(losses) / float(len(losses)), epoch)
    writer.add_scalar('Accuracy', accuracy, epoch)

def validation(epoch):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = Variable(data).cuda(), Variable(target).cuda()
            output = model(data)
            validation_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    writer.add_scalar('Val Loss', validation_loss, epoch)
    writer.add_scalar('Val Accuracy', 100. * correct / len(val_loader.dataset), epoch)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    scheduler.step()
    train(epoch)
    validation(epoch)
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
writer.close()
