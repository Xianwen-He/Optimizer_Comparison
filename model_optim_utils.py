import os
from models import *
import torch_optimizer as optim_library
import torch.optim as optim


def generate_cnn(model_name):
    if model_name == "vgg":
        net = VGG('VGG19')
    elif model_name == 'resnet':
        net = ResNet18()
    elif model_name == 'preact_resnet':
        net = PreActResNet18()
    elif model_name == 'googlenet':
        net = GoogLeNet()
    elif model_name == 'denseset':
        net = DenseNet121()
    elif model_name == 'regnet':
        net = RegNetX_200MF()
    else:
        print('Warning: Default SimpleDLA is applied.')
        net = SimpleDLA()
    return net

def generate_bert(model_name):
    pass

def generate_optimizer(optim_name, optim_params, model):
    if optim_name == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=optim_params['lr'],
                              momentum=optim_params['momentum'],
                              weight_decay=optim_params['weight_decay'])
    elif optim_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                                lr=optim_params['lr'],
                                betas=optim_params['betas'],
                                eps=optim_params['eps'],
                                weight_decay=optim_params['weight_decay'])
    elif optim_name == 'ADAHESSIAN':
        optimizer = optim_library.Adahessian(model.parameters(), 
                                             lr = optim_params['lr'],
                                             betas = optim_params['betas'],
                                             eps = optim_params['eps'],
                                             weight_decay = optim_params['weight_decay'],
                                             hessian_power = optim_params['hessian_power'])
    elif optim_name == 'AdaMod':
        optimizer = optim_library.AdaMod(model.parameters(),
                                         lr = optim_params['lr'],
                                         betas = optim_params['betas'],
                                         beta3 = optim_params['beta3'],
                                         eps = optim_params['eps'],
                                         weight_decay = optim_params['weight_decay'])
    elif optim_name == 'AdamP':
        optimizer = optim_library.AdamP(model.parameters(),
                                         lr = optim_params['lr'],
                                         betas = optim_params['betas'],
                                         eps = optim_params['eps'],
                                         weight_decay = optim_params['weight_decay'],
                                         delta = optim_params['delta'],
                                         wd_ratio = optim_params['wd_ratio'])
    # elif optim_name == '':  # more choices could be put here
    #     pass
    else:
        raise ValueError("Invalid Optimizer.")

    return optimizer