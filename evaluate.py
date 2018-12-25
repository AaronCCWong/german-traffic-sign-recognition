from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets

from data import initialize_data # data.py in the same folder
from base_model import BaseNet
from deep_model import DeepNet
from resnet_model import ResNet

parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='gtsrb_kaggle.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--network', choices=['base', 'deepbase', 'resnet'], default='base',
                    help='model to use (default: base')

args = parser.parse_args()

from data import base_data_transforms, validation_data_transforms

if args.model == 'base':
    model = BaseNet()
    transforms = base_data_transforms
elif args.model == 'deepbase':
    model = DeepNet()
    transforms = base_data_transforms
elif args.model == 'resnet':
    model = ResNet()
    transforms = validation_data_transforms

state_dict = torch.load(args.model)
model.load_state_dict(state_dict)
model.eval()
model.cuda()

test_dir = args.data + '/test_images'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(args.outfile, "w")
output_file.write("Filename,ClassId\n")
with torch.no_grad():
    for f in tqdm(os.listdir(test_dir)):
        if 'ppm' in f:
            data = transforms(pil_loader(test_dir + '/' + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            data = Variable(data).cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]

            file_id = f[0:5]
            output_file.write("%s,%d\n" % (file_id, pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle '
      'competition at https://www.kaggle.com/c/nyu-cv-fall-2018/')



