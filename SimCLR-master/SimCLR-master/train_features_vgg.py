
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import time


from models import *
from utils import *
from data import *
from loss import *

parser = argparse.ArgumentParser(description='SIMCLR')

parser.add_argument('--uid', type=str, default='SimCLR',
                    help='Staging identifier (default: SimCLR)')
parser.add_argument('--dataset-name', type=str, default='CIFAR10C',
                    help='Name of dataset (default: CIFAR10C')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Path to dataset (default: data')
parser.add_argument('--feature-size', type=int, default=128,
                    help='Feature output size (default: 128')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input training batch-size')
parser.add_argument('--accumulation-steps', type=int, default=4, metavar='N',
                    help='Gradient accumulation steps (default: 4')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of training epochs (default: 150)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=1e-6, action="store", type=float,
                    help='Learning rate decay (default: 1e-6')
parser.add_argument('--tau', default=0.5, type=float,
                    help='Tau temperature smoothing (default 0.5)')
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')
parser.add_argument('--load-model', type=str, default=None,
                    help='Load model to resume training for (default None)')
parser.add_argument('--device-id', type=int, default=0,
                    help='GPU device id (default: 0')

args = parser.parse_args()

# Set cuda
use_cuda = not args.no_cuda and torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda")
    torch.cuda.set_device(args.device_id)
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")

# Setup tensorboard
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Setup asset directories
if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('runs'):
    os.makedirs('runs')

# Logger
if use_tb:
    logger = SummaryWriter(comment='_' + args.uid + '_' + args.dataset_name)

# dataset
train_root_dirs = ['/home/ps/zhencunjiang/US_sr/train_datasets/breast/high_quality', '/home/ps/zhencunjiang/US_sr/train_datasets/breast/low_quality',
                   '/home/ps/zhencunjiang/US_sr/train_datasets/kidney/high_quality',
                   '/home/ps/zhencunjiang/US_sr/train_datasets/kidney/low_quality',
                   '/home/ps/zhencunjiang/US_sr/train_datasets/liver/high_quality',
                   '/home/ps/zhencunjiang/US_sr/train_datasets/liver/low_quality'
                   ]
test_root_dirs = ['/home/ps/zhencunjiang/US_sr/train_datasets/carotid/high_quality', '/home/ps/zhencunjiang/US_sr/train_datasets/carotid/low_quality']
train_transforms = cifar_train_transforms()
test_transforms = cifar_test_transforms()
train_dataset = CustomDataset(train_root_dirs, train_transforms)
test_dataset = CustomDataset(test_root_dirs, test_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# if args.dataset_name == 'CIFAR10C':
#     in_channels = 3
#     # Get train and test loaders for dataset
#     train_transforms = cifar_train_transforms()
#     test_transforms = cifar_test_transforms()
#     target_transforms = None
#
#     loader = Loader(args.dataset_name, args.data_dir, True, args.batch_size, train_transforms, test_transforms, target_transforms, use_cuda)
#     train_loader = loader.train_loader
#     test_loader = loader.test_loader


# train validate
def train_validate(model, loader, optimizer, is_train, epoch, use_cuda):

    loss_func = contrastive_loss(tau=args.tau)

    data_loader = loader
    if is_train:
        model.train()
        model.zero_grad()
    else:
        model.eval()

    desc = 'Train' if is_train else 'Validation'

    total_loss = 0.0

    tqdm_bar = tqdm(data_loader)
    for i, (x_i, x_j) in enumerate(tqdm_bar):

        x_i = x_i.cuda() if use_cuda else x_i
        x_j = x_j.cuda() if use_cuda else x_j

        z_i = model(x_i)
        z_j = model(x_j)
        print(z_i.shape)

        loss = loss_func(z_i, z_j)
        loss /= args.accumulation_steps

        if is_train:
            loss.backward()

        if (i + 1) % args.accumulation_steps == 0 and is_train:
            optimizer.step()
            model.zero_grad()

        total_loss += loss.item()

        tqdm_bar.set_description('{} Epoch: [{}] Loss: {:.4f}'.format(desc, epoch, loss.item()))

    return total_loss / (len(data_loader.dataset))


def execute_graph(model, loader1,loader2, optimizer, scheduler, epoch, use_cuda):
    t_loss = train_validate(model, loader1, optimizer, True, epoch, use_cuda)
    v_loss = train_validate(model, loader2, optimizer, False, epoch, use_cuda)

    scheduler.step(v_loss)

    if use_tb:
        logger.add_scalar(log_dir + '/train-loss', t_loss, epoch)
        logger.add_scalar(log_dir + '/valid-loss', v_loss, epoch)

    return v_loss

import  torchvision
import torch.nn.functional as F
VGG_PRETRAIN_PATH = "/home/ps/zhencunjiang/US_sr/SimCLR-master/SimCLR-master/vgg19model.pth"
class vgg19(nn.Module):
    def __init__(self,pretrained=False):
       super().__init__()
       self.model= torchvision.models.vgg19()
       state_dict = torch.load(
           VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage
       )
       self.model.load_state_dict(state_dict)

       self.model.classifier  = nn.Sequential(
           nn.Linear(25088, 512),
           nn.BatchNorm1d(512),
           nn.ReLU(),
           nn.Linear(512, 128)
       )
    def forward(self,x):
        x=self.model(x)
        return F.normalize(x)
model = vgg19().cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay_lr)
scheduler = ExponentialLR(optimizer, gamma=args.decay_lr)


# Main training loop
best_loss = np.inf

# Resume training
if args.load_model is not None:
    if os.path.isfile(args.load_model):
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_loss = checkpoint['val_loss']
        epoch = checkpoint['epoch']
        print('Loading model: {}. Resuming from epoch: {}'.format(args.load_model, epoch))
    else:
        print('Model: {} not found'.format(args.load_model))

for epoch in range(args.epochs):
    v_loss = execute_graph(model, train_loader,test_loader, optimizer, scheduler, epoch, use_cuda)

    if v_loss < best_loss:
        best_loss = v_loss
        print('Writing model checkpoint')
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)

        torch.save(model.state_dict(), '/home/ps/zhencunjiang/US_sr/SimCLR-master/SimCLR-master/USPLOSS.pth')


# TensorboardX logger
logger.close()

# save model / restart training
