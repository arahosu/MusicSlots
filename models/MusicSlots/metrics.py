import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex

import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

import time

def _safe_divide(num, denom):
    denom[denom == 0.0] = 1
    num = num if num.is_floating_point() else num.float()
    denom = denom if denom.is_floating_point() else denom.float()
    return num / denom

def ARI(true_masks, pred_masks, foreground_only=True):

    ari = []

    for true_mask, pred_mask in zip(true_masks, pred_masks):
        n_true_groups, height, width = true_mask.shape
        true_mask_perm = torch.permute(true_mask, (1, 2, 0)) # (H, W, N)
        true_mask = true_mask_perm.argmax(dim=-1).flatten() # (H * W)

        n_pred_groups, _, _ = pred_mask.shape
        pred_mask_perm = torch.permute(pred_mask, (1, 2, 0)) # (H, W, N)
        pred_mask = pred_mask_perm.argmax(dim=-1).flatten() # (H * W)

        if foreground_only:
            pred_mask = pred_mask[true_mask > 0]
            true_mask = true_mask[true_mask > 0]
        score = adjusted_rand_score(
            true_mask.cpu().numpy(), pred_mask.cpu().numpy())
        ari.append(score)
    
    return sum(ari)/len(ari)

def best_overlap_mse(gts, preds, return_idx = False):
    batch_metric = []
    best_overlap_idx = []

    with torch.no_grad():
        for gt, pred in zip(gts, preds):
            # gt shape: num_notes * height * width
            # pred shape: num_slots * height * width

            gt = gt.unsqueeze(dim=1) # num_notes * 1 * heigth * width 
            pred = pred.unsqueeze(dim=0) # 1 * num_slots * height * width
            
            mse_matrix = torch.square(gt - pred).view(gt.size(0), pred.size(1), -1).mean(-1) # num_notes * num_slots
            row, col = linear_sum_assignment(
                mse_matrix.cpu().numpy(), maximize=False)
            metric = torch.mean(mse_matrix[row, col])
            batch_metric.append(metric.unsqueeze(0))
            
            if return_idx:
                best_overlap_idx.append(col)

    if return_idx:
        return torch.mean(torch.cat(batch_metric)), best_overlap_idx
    return torch.mean(torch.cat(batch_metric))

def best_overlap_iou(gts, preds, return_idx = False):
    batch_metric = []
    best_overlap_idx = []

    with torch.no_grad():
        for gt, pred in zip(gts, preds):
            # gt shape: num_notes * height * width
            # pred shape: num_slots * height * width

            gt = gt.unsqueeze(dim=1) # num_notes * 1 * heigth * width 
            pred = pred.unsqueeze(dim=0) # 1 * num_slots * height * width

            overlap = torch.logical_and(gt, pred).view(gt.size(0), pred.size(1), -1).sum(-1)
            union = torch.logical_or(gt, pred).view(gt.size(0), pred.size(1), -1).sum(-1)

            iou_matrix = _safe_divide(overlap, union)

            row, col = linear_sum_assignment(
                iou_matrix.cpu().numpy(), maximize=True)
            best_ov_iou = torch.mean(iou_matrix[row, col])
            batch_metric.append(best_ov_iou.unsqueeze(0))

            if return_idx:
                best_overlap_idx.append(col)
    
    if return_idx:
        return torch.mean(torch.cat(batch_metric)), best_overlap_idx
    return torch.mean(torch.cat(batch_metric))
    

