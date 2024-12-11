# reference code: https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
import math
import os
import random
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from tqdm import tqdm
from imbalance_data.lt_data import LT_Dataset
from losses.loss import *
from opts import parser
import warnings
from torch.nn import Parameter
import torch.nn.functional as F
from utils.util import *
from utils.randaugment import rand_augment_transform
import utils.moco_loader as moco_loader
import models.model as net

from train.train import *
from train.validate import *

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

best_acc1 = 0

def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.arch, args.loss_fn, args.train_rule, args.data_aug, str(args.imb_factor),
         str(args.rand_number),
         str(args.mixup_prob), args.exp_str])
    prepare_folders(args)
        
    if args.cos:
        print("use cosine LR")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda


    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 1000 if args.dataset == 'imgnet' else 8142
    if args.loss_fn == 'ride':
        if args.arch == 'resnet50':
            model = net.resnet50(num_classes)
        elif args.arch == 'resnext50':
            model = net.resnext50(num_classes)
    elif args.loss_fn == 'ncl':
        pass
    elif args.loss_fn == 'bcl':
        model = net.BCLModel(num_classes=num_classes, name=args.arch)
    else:
        if args.arch == 'resnet50':
            model = net.resnet50(num_classes)
        elif args.arch == 'resnext50':
            model = net.resnext50(num_classes)
        
    if hasattr(model, 'backbone'):
        num_ftrs = model.backbone.fc.in_features
        if args.use_norm:
            model.backbone.fc = NormedLinear(num_ftrs, num_classes)
        else:
            model.backbone.fc = nn.Linear(num_ftrs, num_classes)
    else:
        num_ftrs = model.fc.in_features
        if args.use_norm:
            model.fc = NormedLinear(num_ftrs, num_classes)
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.dataset == 'imgnet': 
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.dataset == 'inat':
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
        
    rgb_mean = (0.485, 0.456, 0.406)
    
    ra_params = dict(translate_const=int(224 * 0.45),
                     img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

    augmentation_randncls = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    augmentation_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        normalize
    ])
    if args.loss_fn == 'bcl':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim), transforms.Compose(augmentation_sim)]
    else:
        transform_train = transforms.Compose(augmentation_randncls)
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    if args.dataset == 'imgnet':
        root = 'datasets/imagenet-LT'
        train_config = 'config/ImageNet/ImageNet_LT_train.txt'
        valid_config = 'config/ImageNet/ImageNet_LT_test.txt'
    elif args.dataset == 'inat':
        root = 'datasets/iNaturalist'
        train_config = 'config/iNaturalist/iNaturalist18_train.txt'
        valid_config = 'config/iNaturalist/iNaturalist18_val.txt'
    
    train_dataset = LT_Dataset(root, train_config, transform_train)
    val_dataset = LT_Dataset(root, valid_config, transform_val)
    
    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == 1000 if args.dataset == 'imgnet' else 8142
    args.num_class = num_classes

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    train_cls_num_list = np.array(cls_num_list)
    
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print("data loaders loaded")
    
    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()
    start_time = time.time()
    print("Training started!")
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    if 'CMO' in args.data_aug or args.train_rule == 'CRT':
        cls_weight = 1.0 / (np.array(cls_num_list))
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        labels = train_loader.dataset.targets
        samples_weight = np.array([cls_weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(labels), replacement=True)
        weighted_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=sampler)
    else:
        weighted_trainloader = None
    
    
    train = get_train_fn(args)
    validate = get_valid_fn(args)
    
    test_accs = []

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    print('cls num list:')
    print(cls_num_list)
    N_SAMPLES_PER_CLASS = np.array(cls_num_list)
    train_criterion = get_loss(args, N_SAMPLES_PER_CLASS)
    criterion = nn.CrossEntropyLoss() # For test, validation
    few_best, med_best, many_best = 0, 0, 0
    for epoch in range(args.start_epoch, args.epochs):
        print ('START!!')

        lr = adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(args, train_loader, model, optimizer, train_criterion, epoch, weighted_trainloader)
        
        test_loss, test_acc, test_cls = validate(args, val_loader, model, criterion, N_SAMPLES_PER_CLASS, num_class=args.num_class, mode='test Valid')
    
        # remember best acc@1 and save checkpoint
        is_best = test_acc > best_acc1
        if is_best:          
            best_acc1 = test_acc
            many_best = test_cls[0]
            med_best = test_cls[1]
            few_best = test_cls[2]

        test_accs.append(test_acc)

        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
        }, is_best, epoch + 1)

    end_time = time.time()

    print("It took {} to execute the program".format(hms_string(end_time - start_time)))
    log_testing.write("It took {} to execute the program".format(hms_string(end_time - start_time)) + '\n')
    log_testing.flush()

if __name__ == '__main__':
    main()
