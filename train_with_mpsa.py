import os
# Fix for Deterministic algorithms error
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import project components
# We import directly from the project's modules
from models.build import build_models
from setup import config, log
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.eval import *
from utils.info import *
from main import train_one_epoch, valid, loss_in_iters

# 1. Define a Custom Dataset class for your CSV files
class CustomCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Based on your example: columns are 'img_root' and 'label'
        # 'img_root' contains the full path or relative path to the image
        self.img_paths = self.df['img_root'].values
        self.labels = self.df['label'].values
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image if failed
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        label = int(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 2. Setup Data Loaders
def get_custom_loaders(config, train_csv, val_csv):
    img_size = config.data.img_size
    
    # Simple transforms (you can match project's ones in build_transforms)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = CustomCSVDataset(train_csv, train_transform)
    val_dataset = CustomCSVDataset(val_csv, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(np.unique(train_dataset.labels))
    return train_loader, val_loader, num_classes

# 3. Custom Training Logic
def run_custom_training():
    # --- PLEASE SET YOUR CSV PATHS HERE ---
    TRAIN_CSV = r'D:\HuLintao\dataset\isic2018_train.csv' 
    VAL_CSV = r'D:\HuLintao\dataset\isic2018_test.csv'
    # ----------------------------------

    # Check if files exist
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        log.error(f"CSV files not found at {TRAIN_CSV} or {VAL_CSV}. Please check paths in train_with_mpsa.py")
        return

    # Initialize data
    train_loader, val_loader, num_classes = get_custom_loaders(config, TRAIN_CSV, VAL_CSV)
    log.info(f"Loaded {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")
    log.info(f"Detected {num_classes} classes.")

    # Build MPSA model
    model = build_models(config, num_classes)
    model.to(config.device)
    
    optimizer = build_optimizer(config, model, False)
    scheduler = build_scheduler(config, optimizer, len(train_loader))
    
    # Loss scaling for AMP
    from timm.utils import NativeScaler
    loss_scaler = NativeScaler()
    
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(config.train.epochs):
        # Use project's train_one_epoch
        train_acc = train_one_epoch(config, model, criterion, train_loader, optimizer, 
                                   epoch, scheduler, loss_scaler, None, None)
        
        # Use project's valid
        accuracy, loss = valid(config, model, val_loader, epoch, train_acc, None, False)
        
        if accuracy > best_acc:
            best_acc = accuracy
            log.info(f"*** New Best Accuracy: {best_acc:.2f}% at Epoch {epoch+1} ***")
            # Save the model
            save_path = os.path.join(config.data.log_path, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            log.info(f"Saved to {save_path}")

if __name__ == "__main__":
    # Ensure you use the correct config (isic.yaml)
    # Run command: python train_with_mpsa.py
    # (The config is loaded from setup.py, which defaults to what's in setup.py's cfg_file)
    run_custom_training()
