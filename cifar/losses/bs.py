import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.dbm import DBMargin

class BS(nn.Module):
    def __init__(self, args, num_class_list):
        super().__init__()
        dist = torch.from_numpy(np.array(num_class_list)).float().cuda()
        self.prob = dist / sum(dist)
        self.log_prior = torch.log(self.prob).unsqueeze(0)
        self.use_norm = args.use_norm
        if self.use_norm:
            self.dbm = DBMargin(args, num_class_list)
    def forward(self, logits, targets, epoch=None, reduction='mean'):
        if self.use_norm:
            logits = self.dbm(logits, targets)
        adjusted_logits = logits + self.log_prior
        return F.cross_entropy(adjusted_logits, targets, reduction = reduction)
        
        
        # targets = F.one_hot(targets, num_classes=logits.size(1))
        # logits = logits + torch.log(self.prob.view(1, -1).expand(logits.shape[0], -1)).cuda()
        
        # if reduction == 'none':
        #     return -(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))
        # else:
        #     return -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))