"""
Robust, stand-alone training script with:
- Checkpointing (last_epoch.pt, epoch_X.pt)
- Resuming from last_epoch.pt
- Best model saving (best_epoch.pt)
- Early Stopping
- Metrics history logging (training_history.csv)
- Final testing on test set
- Plotting of training/validation curves
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_score, accuracy_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import yaml
import argparse
import matplotlib.pyplot as plt

# --- Model & Data ---
# Note: These must be importable, so they need to be in src/
# (Assuming they are in src/datasets.py and src/models.py)

try:
    from src.datasets import PatchDataset
    from src.models import ResNet50Fine, ViTModel
except ImportError:
    print("Warning: Could not import from src. Running standalone.")
    # Define dummy classes for environments where src isn't in path
    # This can happen in some notebook setups.
    
    from torch.utils.data import Dataset
    from torchvision.models import resnet50, ResNet50_Weights
    from PIL import Image

    class PatchDataset(Dataset):
        def __init__(self, csv_file, img_dir, transform=None):
            self.data_frame = pd.read_csv(csv_file)
            self.img_dir = img_dir
            self.transform = transform
        def __len__(self):
            return len(self.data_frame)
        def __getitem__(self, idx):
            img_name, label = self.data_frame.iloc[idx, 0], self.data_frame.iloc[idx, 1]
            image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, img_name

    class ResNet50Fine(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        def forward(self, x):
            return self.backbone.x(x)

def load_model(cfg):
    model_config = cfg.get('model', {})
    model_type = model_config.get('type', 'resnet')
    num_classes = model_config.get('num_classes', 2)
    
    if model_type == 'resnet':
        model = ResNet50Fine(num_classes=num_classes)
    else:
        raise NotImplementedError("ViT model loading not implemented.")
    return model

# --- Trainer Class ---

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # 1. Setup Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"--- [Trainer] Using device: {self.device} ---")
        
        # 2. Setup Configs
        self.data_cfg = cfg.get('data', {})
        self.train_cfg = cfg.get('training', {})
        self.outdir = self.train_cfg.get('outdir', 'experiments/default')
        os.makedirs(self.outdir, exist_ok=True, mode=0o777)

        # 3. Setup Checkpoint Paths
        self.last_ckpt_path = os.path.join(self.outdir, 'last_epoch.pt')
        self.best_ckpt_path = os.path.join(self.outdir, 'best_epoch.pt')
        self.history_csv_path = os.path.join(self.outdir, 'training_history.csv')
        self.plot_path = os.path.join(self.outdir, 'training_plot.png')

        # 4. Setup DataLoaders
        img_size = self.data_cfg.get('img_size', 224)
        train_t = T.Compose([T.RandomResizedCrop(img_size), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        val_t = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

        # Train Loader
        self.train_ds = PatchDataset(self.data_cfg['train_csv'], self.data_cfg['img_dir'], transform=train_t)
        self.train_loader = DataLoader(self.train_ds, batch_size=self.train_cfg.get('batch_size', 32), shuffle=True, num_workers=0)
        print(f"✅ Created training loader with {len(self.train_ds)} samples")

        # Validation Loader
        self.val_ds = PatchDataset(self.data_cfg['val_csv'], self.data_cfg['img_dir'], transform=val_t)
        self.val_loader = DataLoader(self.val_ds, batch_size=self.train_cfg.get('batch_size', 32), shuffle=False, num_workers=0)
        print(f"✅ Created validation loader with {len(self.val_ds)} samples")
        
        # Test Loader
        self.test_ds = PatchDataset(self.data_cfg['test_csv'], self.data_cfg['img_dir'], transform=val_t)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.train_cfg.get('batch_size', 32), shuffle=False, num_workers=0)
        print(f"✅ Created test loader with {len(self.test_ds)} samples")

        # 5. Setup Model, Optimizer, Scheduler
        self.model = load_model(cfg)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.train_cfg.get('lr', 0.0001))
        self.total_epochs = self.train_cfg.get('epochs', 20)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.total_epochs)

        # 6. Setup State for Resuming & Early Stopping
        self.start_epoch = 0
        self.best_val_acc = -1.0
        self.epochs_no_improve = 0
        self.patience = self.train_cfg.get('early_stopping_patience', 10)
        self.history = []

    def load_checkpoint(self):
        if os.path.exists(self.last_ckpt_path):
            print(f"🔄 Resuming training from checkpoint: {self.last_ckpt_path}")
            checkpoint = torch.load(self.last_ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint.get('best_val_acc', -1.0) # Use .get for backward compatibility
            self.epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            print(f"✅ Resumed from epoch {self.start_epoch}. Best val_acc so far: {self.best_val_acc:.4f}")
        
        if os.path.exists(self.history_csv_path):
            self.history = pd.read_csv(self.history_csv_path).to_dict('records')

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'epochs_no_improve': self.epochs_no_improve
        }
        # Save last_epoch.pt (overwritten)
        torch.save(checkpoint, self.last_ckpt_path)
        
        # Save individual epoch file
        epoch_save_path = os.path.join(self.outdir, f'epoch_{epoch+1}.pt')
        torch.save(checkpoint, epoch_save_path)
        # print(f"Saved checkpoint to {epoch_save_path}")

    def save_history_to_csv(self):
        pd.DataFrame(self.history).to_csv(self.history_csv_path, index=False)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss, all_labels, all_preds = 0.0, [], []
        loop = tqdm(self.train_loader, desc=f'Train E{epoch+1}/{self.total_epochs}', leave=True)
        
        for imgs, labels, _ in loop:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            preds_logits = self.model(imgs)
            loss = self.criterion(preds_logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, p = preds_logits.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(p.cpu().numpy())
            loop.set_postfix(loss=loss.item())
        
        loss = running_loss / len(self.train_loader)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        return loss, acc, prec

    def validate(self, epoch, loader):
        self.model.eval()
        running_loss, all_labels, all_preds = 0.0, [], []
        desc = f'Validate E{epoch+1}/{self.total_epochs}' if loader == self.val_loader else 'Testing'
        loop = tqdm(loader, desc=desc, leave=True)
        
        with torch.no_grad():
            for imgs, labels, _ in loop:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                preds_logits = self.model(imgs)
                loss = self.criterion(preds_logits, labels)
                running_loss += loss.item()
                _, p = preds_logits.max(1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(p.cpu().numpy())
                loop.set_postfix(loss=loss.item())
        
        loss = running_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        return loss, acc, prec

    def run_training(self):
        print(f"🚀 Starting training for {self.total_epochs} epochs...")
        for epoch in range(self.start_epoch, self.total_epochs):
            # 1. Train
            train_loss, train_acc, train_prec = self.train_epoch(epoch)
            
            # 2. Validate
            val_loss, val_acc, val_prec = self.validate(epoch, self.val_loader)
            
            print(f"Epoch {epoch+1} Results: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 3. Log History
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss, 'train_acc': train_acc, 'train_prec': train_prec,
                'val_loss': val_loss, 'val_acc': val_acc, 'val_prec': val_prec
            })
            self.save_history_to_csv()

            # 4. Step Scheduler
            self.scheduler.step()

            # 5. Checkpointing & Best Model
            self.save_checkpoint(epoch) # Save last_epoch.pt and epoch_X.pt
            
            if val_acc > self.best_val_acc:
                print(f"🎉 New best validation accuracy: {val_acc:.4f} (was {self.best_val_acc:.4f}). Saving best model...")
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.best_ckpt_path) # Save best_epoch.pt
            else:
                self.epochs_no_improve += 1
                print(f"Validation accuracy did not improve. Patience: {self.epochs_no_improve}/{self.patience}")

            # 6. Early Stopping
            if self.epochs_no_improve >= self.patience:
                print(f"🛑 Early stopping triggered at epoch {epoch+1} after {self.patience} epochs with no improvement.")
                break
        print("🏁 Training finished.")

    def run_testing(self):
        print("\n--- Running Final Test ---")
        if not os.path.exists(self.best_ckpt_path):
            print("❌ No 'best_epoch.pt' model found. Testing with last available model.")
            # Fallback to last checkpoint if best was never saved
            if os.path.exists(self.last_ckpt_path):
                checkpoint = torch.load(self.last_ckpt_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                print("❌ No models found. Cannot run test.")
                return
        else:
            print(f"✅ Loading best model from {self.best_ckpt_path} (Val Acc: {self.best_val_acc:.4f})")
            self.model.load_state_dict(torch.load(self.best_ckpt_path, map_location=self.device))
        
        test_loss, test_acc, test_prec = self.validate(epoch=0, loader=self.test_loader)
        
        print("\n" + "="*30)
        print("🎯 FINAL TEST RESULTS 🎯")
        print(f"     Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print("="*30)

    def plot_history(self):
        if not self.history:
            print("No history to plot.")
            return
            
        df = pd.DataFrame(self.history)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f"Training History: {self.train_cfg.get('experiment_name')}")

        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
        ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
        ax2.plot(df['epoch'], df['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plot_path)
        print(f"📈 Saved training plot to {self.plot_path}")
        plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized Training Script")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Initialize Trainer
    trainer = Trainer(cfg)
    
    # Load checkpoint if it exists
    trainer.load_checkpoint()
    
    # Run Training
    trainer.run_training()
    
    # Plot History
    trainer.plot_history()
    
    # Run Final Test
    trainer.run_testing()

print("✅ src/train.py has been updated with full training, checkpointing, resuming, and testing logic.")
