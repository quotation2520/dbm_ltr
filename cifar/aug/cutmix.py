import torch
import numpy as np
import torch.nn.functional as F

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(data_f, data_b, alpha=1.):
    lam = np.random.beta(alpha, alpha)
    lam = np.max([lam, 1. - lam])
    bbx1, bby1, bbx2, bby2 = rand_bbox(data_f.size(), lam)
    data_b[:, :, bbx1:bbx2, bby1:bby2] = data_f[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1. -((bbx2 - bbx1) * (bby2 - bby1) / (data_f.size()[2] * data_f.size()[3]))
    
    return data_b, torch.tensor(lam)

def mixup(data_f, data_b, alpha=1.):
    lam = np.random.beta(alpha, alpha)
    lam = np.max([lam, 1. - lam])
    data = data_f * lam + data_b * (1 - lam)
    return data, torch.tensor(lam)

def resizemix(data_f, data_b, alpha=1.):
    lam = np.random.beta(alpha, alpha)
    lam = np.max([lam, 1. - lam])
    bbx1, bby1, bbx2, bby2 = rand_bbox(data_f.size(), lam)
    lam = 1. -((bbx2 - bbx1) * (bby2 - bby1) / (data_f.size()[2] * data_f.size()[3]))
    if lam < 1:
        data_b[:, :, bbx1:bbx2, bby1:bby2] = F.interpolate(data_f, size=((bbx2-bbx1, bby2-bby1)), mode='bilinear')
    
    return data_b, torch.tensor(lam)