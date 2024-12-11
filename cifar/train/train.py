from train.train_fn.base import train_base
from train.train_fn.ncl import train_ncl
from train.train_fn.bcl import train_bcl

def get_train_fn(args):
    if args.loss_fn == 'ncl':
        return train_ncl
    elif args.loss_fn == 'bcl':
        return train_bcl
    else:
        return train_base
