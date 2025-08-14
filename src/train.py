import os
import yaml
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.datasets import PatchDataset
from src.models import ResNet50Fine, ViTModel
from src.metrics import iou_from_masks
from src.xai import XAIProcessor

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device('cpu')

def build_transforms(img_size=224):
    train_t = T.Compose([
        T.RandomResizedCrop(img_size, scale(0.8,1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    val_t = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    return train_t, val_t

def load_model(cfg):
    if cfg['model']['type'] == 'resnet':
        model = ResNet50Fine(num_classes=cfg['model']['num_classes'])
    else:
        model = ViTModel(model_name=cfg['model'].get('vit_name', 'vit_base_patch16_224'), num_classes=cfg['model']['num_classes'])
    return model

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device()
        os.makedirs(cfg['training']['outdir'], exist_ok=True)

        train_t, val_t = build_transforms(cfg['data'].get('img_size', 224))
        self.train_ds = PatchDataset(cfg['data']['train_csv'], cfg['data']['img_dir'], transform=train_t)
        self.val_ds = PatchDataset(cfg['data']['val_csv'], cfg['data']['img_dir'], transform=val_t)

        self.train_loader = DataLoader(self.train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        
        self.model = load_model(cfg)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training'].get('weight_decay',1e-4))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1,cfg['training']['epochs']), eta_min=1e-6)
        
        self.xai_reg = cfg['training'].get('use_xai_reg', False)
        if self.xai_reg:
            self.xai_processor = XAIProcessor(self.model, self.device)
            self.masks_root = cfg['data'].get('masks_root', None)
            if self.masks_root is None:
                print('Warning: XAI regularizer enabled but masks_root not provided. Turning off XAI reg.')
                self.xai_reg = False
        self.best_val = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        tk = tqdm(self.train_loader, desc=f'Train E{epoch}')
        for imgs, labels, filenames in tk:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            preds = self.model(imgs)
            loss = self.criterion(preds, labels)

            if self.xai_reg:
                with torch.no_grad():
                    pass
                reg_losses = []
                for i in range(imgs.size(0)):
                    fname = filenames[i]
                    mask_path = os.path.join(self.masks_root, fname.replace('.png','_mask.png'))

                    if not os.path.exists(mask_path):
                        continue
                    pil = Image.open(os.path.join(self.train_ds.img_dir,fname)).convert('RGB')
                    try:
                        heat = self.xai_processor.gradcam(pil)
                    except Exception as e:
                        continue
                    mask = np.array(Image.open(mask_path).convert('L').resize((heat.shape[1],heat.shape[0])))
                    mask = (mask>127).astype(np.float32)
                    iou = iou_from_masks(heat, mask, thresh=0.5)
                    reg = 1.0 - iou
                    reg_losses.append(reg)
                if len(reg_losses) > 0:
                    reg_loss = float(np.mean(reg_losses))
                    loss = loss + self.cfg['training'].get('xai_reg_lambda',1.0) * reg_loss
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            tk.set_postfix(loss=running_loss/(tk.n+1))

        return running_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        correct, total = 0, 0
        losses = []
        with torch.no_grad():
            for imgs, labels, _ in tqdm(self.val_loader, desc=f'Val E{epoch}'):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                preds = self.model(imgs)
                loss = self.criterion(preds, labels)
                losses.append(loss.item())
                _, p = preds.max(1)
                correct += (p==labels).sum().item()
                total += labels.size(0)
        acc = correct/total if total>0 else 0.0
        return np.mean(losses), acc
    
    def fit(self):
        epochs = self.cfg['training']['epochs']
        for epoch in range(1, epochs+1):
            t0 = time.time()
        train_loss = self.train_epoch(epoch)
        val_loss, val_acc = self.validate(epoch)
        self.scheduler.step()
        print(f'Epoch {epoch} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Val acc {val_acc:.4f}')
        if val_acc > self.best_val:
            self.best_val = val_acc
            save_path = os.path.join(self.cfg['training']['outdir'], f'best_{self.cfg['model']['type']}.pt')
            torch.save(self.model.state_dict(), save_path)
            print('Saved', save_path)

if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/default.yaml'))
    trainer = Trainer(cfg)
    trainer.fit()