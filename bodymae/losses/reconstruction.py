import torch
import torch.nn as nn
import torch.nn.functional as F
from bodymae.losses.chamfer import ChamferDistanceL2


class AngularLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        pred_u = F.normalize(pred, dim=-1, eps=self.eps)
        gt_u = F.normalize(gt, dim=-1, eps=self.eps)
        dot = (pred_u * gt_u).sum(dim=-1).clamp(-1.0 + self.eps, 1.0 - self.eps)
        angle = torch.acos(dot) / torch.pi
        return angle.mean()




class NormalLoss(nn.Module):
    def __init__(self, eps=1e-7, alpha_unit=0.0):
        super().__init__()
        self.eps = eps
        self.alpha_unit = alpha_unit

    def forward(self, pred, gt):

        pred_unit = F.normalize(pred, dim=-1, eps=self.eps)
        gt_unit = F.normalize(gt, dim=-1, eps=self.eps)
        cos = (pred_unit * gt_unit).sum(dim=-1).clamp(-1.0 + self.eps, 1.0 - self.eps)

        loss = (1.0 - cos).clamp_min(0.0) / 2.0

        loss = loss.mean()

        pred_norm = pred.norm(dim=-1).clamp_min(self.eps)
        loss_unit = F.mse_loss(pred_norm, torch.ones_like(pred_norm))
        loss = loss + self.alpha_unit * loss_unit
        return loss




class RepulsionLoss(nn.Module):
    def __init__(self, k=5, h=0.01, eps=1e-12, use_gaussian=False, sigma=0.01):
        super().__init__()
        self.k = k
        self.h = h
        self.eps = eps
        self.use_gaussian = use_gaussian
        self.sigma = sigma

    def forward(self, x_pred):

        D_pred = torch.cdist(x_pred, x_pred) + self.eps
        D_pred.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        d_k, _ = torch.topk(D_pred, k=self.k, largest=False)
        if self.use_gaussian:
            return torch.exp(-(d_k**2) / (self.sigma**2)).mean()


        hinge = F.relu(self.h - d_k)

        return torch.mean((hinge / (self.h + self.eps)) ** 2)


class ChamferLossL2(nn.Module):
    def __init__(self, eps=1e-8, use_repulsion=True, k=5, h=0.01, use_gaussian=False, sigma=0.01):
        super().__init__()
        self.eps = eps
        self.use_repulsion = use_repulsion
        self.k = k
        self.h = h
        self.use_gaussian = use_gaussian
        self.sigma = sigma
        self.chamfer_distance = ChamferDistanceL2()

    def forward(self, x_pred, x_gt):
        r = x_gt.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(self.eps).unsqueeze(-1)

        x_gt_unit = x_gt / r
        x_pred_unit = x_pred / r
        chamfer_distance = self.chamfer_distance(x_pred_unit, x_gt_unit)
        chamfer_distance = chamfer_distance / 4.0


        if self.use_repulsion:
            D_pred = torch.cdist(x_pred, x_pred) + self.eps
            D_pred.diagonal(dim1=1, dim2=2).fill_(float("inf"))
            d_k, _ = torch.topk(D_pred, k=self.k, largest=False)
            hinge = F.relu(self.h - d_k)
            repulsion = torch.mean((hinge / (self.h + self.eps)) ** 2)
            return chamfer_distance, repulsion

        return chamfer_distance
