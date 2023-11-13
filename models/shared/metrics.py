import torch

def multilabel_acc(gt, pred):
    gt = gt.long()
    pred = (pred >= 0.5).long()
    return 100 * (torch.amin(torch.eq(gt, pred), dim=[1])).float().mean()

def compute_accuracy(preds, gts):
    """
    Compute the accuracy
    """
    assert len(preds) == len(gts)
    acc = sum([(pred == gt).all()
               for pred, gt in zip(preds, gts)])
    return 100 * float(acc) / len(preds)