from models.models import *
from datasets import idrad_tba
from deploy import train, test

from torch.utils.data import DataLoader
from transforms.transforms import microdoppler_transform

import os
import functools
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils import load_checkpoint, load_json, load_config

# cuda device setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# The dataset can be downloaded from the link in the github
parser = argparse.ArgumentParser(description='Radar Human Localization')
parser.add_argument('--params', default='', type=str)
parser.add_argument('--name', default='mymodel', type=str)
parser.add_argument('--network', default='CLNet', type=str)
parser.add_argument('--targets', default=['target1', 'target2', 'target3', 'target4', 'target5'], nargs='+', type=str) # ,
parser.add_argument('--features', default='microdoppler_synthesis', type=str)
parser.add_argument('--learning_rate', default=10 ** -3, type=float)
parser.add_argument('--batch_size', default=48, type=int)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--max_epochs', default=500, type=int)
parser.add_argument('--test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Initialize network
net = eval(args.network)(input_dim=(1, 45, 205), output_dim=(2,900))
net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()

if args.params != "":
    load_checkpoint(net, None, args.params)

values = dict()
values['microdoppler'] = {'mean':-43898.7272684, 'std':1457.87896807, 'min':-46583.265625, 'max':-29791.0058594}
values['microdoppler_thresholded'] = {'mean':-16987.4060019, 'std':619.551479691, 'min':-17100.0, 'max':-6727.47998047}
values['microdoppler_synthesis'] = {'mean':-16987.4060019, 'std':619.551479691, 'min':-17100.0, 'max':-6227.47998047}

transform = functools.partial(microdoppler_transform, values=values[args.features], standard_scaling=True, preprocessing=True)

config = load_config('./config/data_config.json')

dataset = dict(train=idrad_tba("train", config, transform=transform, in_memory=True),
               valid=idrad_tba("val", config, transform=transform, in_memory=True),
               test=idrad_tba("test", config, transform=transform, in_memory=True))

train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, num_workers=12, shuffle=True)
valid_loader = DataLoader(dataset["valid"], batch_size=args.batch_size, num_workers=12)
test_loader = DataLoader(dataset["test"], batch_size=args.batch_size, num_workers=12)

print("---------")
print("%d samples and %d batches in train set." % (len(dataset['train']), len(train_loader)))
print("%d samples and %d batches in validation set." % (len(dataset['valid']), len(valid_loader)))
print("%d samples and %d batches in test set." % (len(dataset['test']), len(test_loader)))
print("---------")

if not args.test:
    train(net,
          dict(train=train_loader, valid=valid_loader, test=test_loader),
          args.name,
          optimizer=optimizer,
          criterion=criterion,
          max_epochs=args.max_epochs,
          phases=['train', 'valid', 'test'], 
          classlabels=args.targets)

if not args.test:
    print("Testing...")
    net.load_state_dict(torch.load(os.path.join('params', args.name + '.pt')), strict=False)
    test(net, 
        dict(valid=valid_loader), 
        args.name, 
        criterion=criterion, 
        max_epochs=1,
        phases=['test'],
        classlabels=args.targets)
