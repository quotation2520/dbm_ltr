import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Beta

import math

class DBMargin(nn.Module):
    def __init__(self, args, dist):
        super(DBMargin, self).__init__()       
        self.n_classes = len(dist) 
        self.scale = args.cos_scale
        self.use_dbm = args.use_dbm
        
        self.dist = torch.from_numpy(np.array(dist)).float().cuda()
        self.prob = self.dist / sum(self.dist)
        self.margin = ((1 / self.prob ** args.tau) / (1 / self.prob ** args.tau)[-1]).unsqueeze(0) * args.max_margin
        self.lambda_inst = args.lambda_inst
            
    def forward(self, logits, target):    
        if self.use_dbm:
            target_oh = F.one_hot(target, num_classes=self.n_classes)
            incorrect = logits.argmax(dim=-1) != target
            margin_seed = self.margin * target_oh
            margin_cls = margin_seed
            
            cosine = logits
            inst_diff = (1 - cosine) / 2 
            margin_inst = margin_cls * inst_diff * incorrect.unsqueeze(-1)
            margin = margin_cls + self.lambda_inst * margin_inst 
                       
            logits = self.scale * torch.cos(torch.acos(cosine) + margin) 
        else:   # Using cosine classifier without margin
            logits = self.scale * logits
        
        return logits