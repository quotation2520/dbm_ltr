import torch
import torch.optim as optim
from bisect import bisect_right

from losses.bs import BS
from losses.ce_drw import CE_DRW
from losses.ce import CE
from losses.ldam_drw import LDAM_DRW
from losses.ncl import NIL_NBOD
from losses.bcl import BCLLoss

from torch.optim import lr_scheduler


def get_loss(args, N_SAMPLES_PER_CLASS, init_proxy=None):
    if args.loss_fn == 'ce':
        train_criterion = CE(args, N_SAMPLES_PER_CLASS)
    elif args.loss_fn == 'ce_drw':
        train_criterion = CE_DRW(args, cls_num_list=N_SAMPLES_PER_CLASS, reweight_epoch=160)
    elif args.loss_fn == 'bs':
        train_criterion = BS(args, N_SAMPLES_PER_CLASS)
    elif args.loss_fn == 'ldam':
        train_criterion = LDAM_DRW(args, cls_num_list=N_SAMPLES_PER_CLASS, reweight_epoch=999, max_m=0.5, s=30).cuda()
    elif args.loss_fn == 'ldam_drw':
        train_criterion = LDAM_DRW(args, cls_num_list=N_SAMPLES_PER_CLASS, reweight_epoch=160, max_m=0.5, s=30).cuda()
        train_criterion = train_criterion.to(torch.device('cuda'))
    elif args.loss_fn == 'ncl':
        train_criterion = NIL_NBOD(args, N_SAMPLES_PER_CLASS)
    elif args.loss_fn == 'bcl':
        train_criterion = BCLLoss(args, N_SAMPLES_PER_CLASS)
    else:
        raise NotImplementedError
        

    return train_criterion

