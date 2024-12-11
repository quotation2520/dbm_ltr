import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.dbm import DBMargin

class CE(nn.Module):
    def __init__(self, args, cls_num_list, weight=None):
        super().__init__()
        self.weight = weight
        self.use_norm = args.use_norm
        if self.use_norm:
            self.dbm = DBMargin(args, cls_num_list)
    def forward(self, logits, targets, epoch=None, reduction='mean'):
        # targets = F.one_hot(targets, num_classes=logits.size(1))
        # if reduction == 'mean':
        #     return -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))
        # else:
        #     return -(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))
        if self.use_norm:
            logits = self.dbm(logits, targets)
        return F.cross_entropy(logits, targets, weight = self.weight, reduction = reduction)