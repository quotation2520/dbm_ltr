# CMO
# Copyright (c) 2022-present NAVER Corp.
# MIT License

import argparse
import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--root', default='../datasets/', help='dataset directory')
parser.add_argument('--dataset', default='cifar100', help='dataset setting', choices=('cifar100', 'imgnet', 'inat')
)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', help='resnet10 | resnet50 | resnext50')
parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')

parser.add_argument('--loss_fn', type=str, default='ce', help='Loss function for training')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader',
                    choices=('None', 'CBReweight', 'DRW', 'CRT'))
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_steps', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true', help='use cosine LR')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--test_epoch', default=1, type=int,
                    metavar='N', help='Test epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', help='utiliizing nesterov SGD')
parser.add_argument('--cutout', default=True, action='store_true', help='utiliizing cutout')
parser.add_argument('--cmo', action='store_true', help='Utilizing CMO')

# data augmentation setting
parser.add_argument('--data_aug', default="vanilla", type=str, help='data augmentation type',
                    choices=('vanilla', 'CMO', 'CUDA', 'CUDA_CMO'))
parser.add_argument('--aug_prob', default=0.5, type=float, help="augment prob")
parser.add_argument('--sim_type', default='none')

parser.add_argument('--mixup_prob', default=0.5, type=float, help='mixup probability')
parser.add_argument('--start_data_aug', default=3, type=int, help='start epoch for aug')
parser.add_argument('--end_data_aug', default=3, type=int, help='how many epochs to turn off aug')
parser.add_argument('--weighted_alpha', default=1, type=float, help='weighted alpha for sampling probability (q(1,k))')
parser.add_argument('--beta', default=1, type=float, help='hyperparam for beta distribution')
parser.add_argument('--use_randaug', action='store_true')

# etc.
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-p', '--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')

## Difficulty-aware Balancing Margin
parser.add_argument('--use_norm', default=False, action='store_true')
parser.add_argument('--use_dbm', action='store_true', help='Use difficulty-aware balancing margin')
parser.add_argument('--cos_scale', type=float, default=30.0, help='scaling factor')
parser.add_argument('--max_margin', default=0.1, type=float, help='hyperparameter K in paper')
parser.add_argument('--tau', default=1.0, type=float, help='hyperparameter tau in paper')
parser.add_argument('--lambda_inst', default=1.0, type=float, help='lambda for instance-wise margin')
