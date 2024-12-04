'''Train CNN on CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import copy
import time
# from tqdm import tqdm
import pandas as pd
import random

from models import *
from utils import load_from_json
from model_optim_utils import generate_cnn, generate_optimizer



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--json_file', default='./json/cifar10_resnet_sgd.json',
                    type=str, help='path to the file containing arguments')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
print(args)


### Global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hyper_args = load_from_json(args.json_file)
best_acc = 0  # best test accuracy
best_state = None  # state of the model with highest acc on the val set
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_epoch = start_epoch  # epoch achieving the highest acc on the val set
record_test_acc = 0  # test acc for the model to save

### Set random seed
seed = hyper_args["seed"]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
# ensure deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### Data
print('==> Preparing data..')

# data processing for CIFAR10
dataset_args = hyper_args["dataset_params"]
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# randomness control
generator = torch.Generator().manual_seed(seed)

# load training set
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(trainset))  # 80% for training
val_size = len(trainset) - train_size  # 20% for validation
trainset, valset = random_split(trainset, [train_size, val_size], generator=generator)

# wrap in the data loader
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=dataset_args['batch_size'], shuffle=dataset_args['shuffle'],
    generator=generator,
    worker_init_fn=lambda worker_id: random.seed(seed + worker_id))
valloader = torch.utils.data.DataLoader(
    valset, batch_size=dataset_args['batch_size'], shuffle=dataset_args['shuffle'],
    generator=generator,
    worker_init_fn=lambda worker_id: random.seed(seed + worker_id))
testloader = torch.utils.data.DataLoader(
    testset, batch_size=dataset_args['test_batch_size'], shuffle=dataset_args['shuffle'],
    generator=generator,
    worker_init_fn=lambda worker_id: random.seed(seed + worker_id))


### Model and Optimizer
print('==> Building model..')
net = generate_cnn(hyper_args["model"])
net = net.to(device)

if args.resume:
    # Load checkpoint
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(hyper_args['checkpoint'])
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['start_epoch']

# initialize optimizer
criterion = nn.CrossEntropyLoss()
optimizer = generate_optimizer(hyper_args['optimizer'], hyper_args['optimizer_params'], net)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)


### Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward(create_graph=True)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    end_time = time.time()
    print('Training: Loss: %.3f | Acc: %.3f (%d/%d)' % (train_loss/len(trainloader), 100.*correct/total, correct, total))
        
    # report the loss, acc, and time used per epoch
    return train_loss/len(trainloader), 100.*correct/total, end_time-start_time

### validating
def validate(epoch):
    global best_acc, best_state, best_epoch

    net.eval()

    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Validating: Loss: %.3f | Acc: %.3f (%d/%d)' % (val_loss/len(valloader), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc>best_acc and epoch>=start_epoch+3:
        print('Saving..')
        test_loss, test_acc = test(net)  # save the testing information
        state = {
            'net': copy.deepcopy(net.state_dict()),
            'acc': test_acc,
            'loss': test_loss,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, hyper_args['save']+'_epoch{}_acc{:.2f}.pt'.format(epoch, acc))
        best_acc = acc
        best_state = state
        best_epoch = epoch

    return val_loss/len(valloader), acc

### Testing
def test(net):
    global record_test_acc
    
    # print('Testing...')
    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    record_test_acc = 100.*correct/total
    print('Testing: Loss: %.3f | Acc: %.3f (%d/%d)' % (test_loss/len(testloader), record_test_acc, correct, total))
    
    # return the value
    return test_loss/len(testloader), record_test_acc
   


### Training
trace_dicn = {"epoch": [],
              "train_loss": [], "train_acc": [], "train_time": [],
              "val_loss": [], "val_acc": []}
print('==> Training model..')
for epoch in range(start_epoch, start_epoch+hyper_args['epoch']):
    # training and validating
    train_loss, train_acc, train_time = train(epoch)
    val_loss, val_acc = validate(epoch)  # testing included if a better model is obtained

    # save training information
    trace_dicn['epoch'].append(epoch)
    trace_dicn['train_loss'].append(train_loss)
    trace_dicn['train_acc'].append(train_acc)
    trace_dicn['train_time'].append(train_time)
    trace_dicn['val_loss'].append(val_loss)
    trace_dicn['val_acc'].append(val_acc)

    scheduler.step()


### Saving results
print('==> Saving results..')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(best_state, hyper_args['save']+'_epoch{}_acc{:.2f}.pt'.format(best_epoch, record_test_acc))

trace_df = pd.DataFrame(trace_dicn)
trace_df.to_csv(hyper_args['save']+"_epoch{}to{}.csv".format(start_epoch, start_epoch+hyper_args['epoch']),
                 index=False)

print("====ALL SET====")
