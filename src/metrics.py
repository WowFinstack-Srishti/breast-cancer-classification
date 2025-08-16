import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def iou_from_masks(pred_mask, true_mask, thresh=0.5):
    p = (pred_mask >= thresh).astype(np.uint8)
    t = (true_mask >= 0.5).astype(np.uint8)
    inter = np.logical_and(p,t).sum()
    union = np.logical_or(p,t).sum()
    if union == 0:
        return 1.0 if inter==0 else 0.0
    return inter/union

def dice(pred_mask, true_mask, thresh=0.5):
    p = (pred_mask >= thresh).astype(np.uint8)
    t = (true_mask >= 0.5).astype(np.uint8)
    inter = 2 * np.logical_and(p,t).sum()
    denom = p.sum() + t.sum()
    return inter/denom if denom>0 else 1.0

# deletion/insertion tests: simplified implementations
from copy import deepcopy
from PIL import Image
import numpy as np

def deletion_insertion_score(model, img_pil, heatmap, device, metric='auc', steps=10):
    # heatmap: 2D normalized [0,1] same size as model input
    # make a sequence of images with increasing deletion of top-k pixels
    from torchvision.transforms import ToTensor, ToPILImage
    t = ToTensor()
    base = t(img_pil).unsqueeze(0)
    h = heatmap
    h_flat = h.flatten()
    order = np.argsort(-h_flat)
    scores = []
    for s in range(1, steps+1):
        k = int(len(order) * s / steps)
        mask = np.ones_like(h_flat) #descending
        mask[order[:k]] = 0
        mask = mask.reshape(h.shape)
        # apply mask by multiplying channels
        m_tensor = torch.tensor(mask).unsqueeze(0).repeat(1,3,1,1).float()
        img_mod = base * m_tensor
        img_mod = img_mod.to(device)
        with torch.no_grad():
            out = model(img_mod)
            prob = torch.softmax(out, dim=1).cpu().numpy()[0,1]
        scores.append(prob)
    # compute AUC-like area under curve
    auc = np.trapzoid(scores) / len(scores)
    return auc