import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast, GradScaler 
from google.colab import drive

from model import AttentionUNet
from topology_loss import ISROCombinedLoss
from dataset import UniversalRoadDataset

def train_model():
    # 1. Mount Drive to access datasets and save checkpoints
    if not os.path.exists('/content/drive/MyDrive'):
        print("☁️ Connecting to Google Drive...")
        drive.mount('/content/drive')

    DRIVE_SAVE_DIR = '/content/drive/MyDrive/SETU_Checkpoints'
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
    
    # Checkpoint paths
    CHECKPOINT_PATH = os.path.join(DRIVE_SAVE_DIR, 'setu_universal_model_latest.pth')
    
    # ⚠️ UPDATE THIS LINE to point to exactly where your SpaceNet final weights are saved in your Drive!
    FOUNDATION_WEIGHTS = '/content/drive/MyDrive/SETU_Checkpoints/spacenet_unet_model_FINAL.pth' 
    
    EFFECTIVE_BATCH_SIZE = 8 
    ACTUAL_BATCH_SIZE = 1
    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // ACTUAL_BATCH_SIZE
    LEARNING_RATE = 1e-5 
    EPOCHS = 15 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("❌ CRITICAL ERROR: GPU NOT DETECTED. Stop the script and switch Runtime to T4 GPU!")
        return
        
    print(f"🌍 S.E.T.U. Universal Training Engine Engaged: {device.type.upper()}")

    # ISRO Augmentation Pipeline
    isro_transform = A.Compose([
        A.Resize(256, 256), 
        A.RandomFog(fog_coef_range=(0.3, 0.7), p=0.4),
        A.GaussNoise(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5), 
        A.CoarseDropout(num_holes_range=(5, 15), hole_height_range=(10, 40), hole_width_range=(10, 40), fill=0, fill_mask=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    print("🔍 Scanning Google Drive for harvested cities...")
    dataset_dir = '/content/drive/MyDrive/SETU_Datasets'
    datasets_to_merge = []
    
    # Dynamically load every folder that ends in '_fast'
    for folder_name in os.listdir(dataset_dir):
        if folder_name.endswith('_fast'):
            full_path = os.path.join(dataset_dir, folder_name)
            print(f"   --> Loading {folder_name.upper()}...")
            city_data = UniversalRoadDataset(full_path, transform=isro_transform)
            if len(city_data) > 0:
                datasets_to_merge.append(city_data)

    if not datasets_to_merge:
        print("❌ CRITICAL ERROR: No datasets found in SETU_Datasets!")
        return

    # Merge them into a single domain-agnostic dataset
    universal_dataset = ConcatDataset(datasets_to_merge)
    train_loader = DataLoader(universal_dataset, batch_size=ACTUAL_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    print("Initializing Hybrid TransUNet Architecture...")
    model = AttentionUNet(img_ch=3, output_ch=1).to(device)
    criterion = ISROCombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler('cuda')

    # Transfer Learning Logic
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️ Found existing Universal checkpoint! Resuming fine-tuning...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    elif os.path.exists(FOUNDATION_WEIGHTS):
        print(f"🚀 Injecting SpaceNet foundation knowledge...")
        checkpoint = torch.load(FOUNDATION_WEIGHTS, map_location=device, weights_only=True)
        # Handle difference between full checkpoints and raw state dicts
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"❌ CRITICAL ERROR: Foundation weights not found at {FOUNDATION_WEIGHTS}!")
        return

    print(f"Starting Pan-India Transfer Learning on {len(universal_dataset)} total satellite chips...")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad() 
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast('cuda'):
                predictions = model(images)
                loss = criterion(predictions, masks) / ACCUMULATION_STEPS 

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 

            epoch_loss += loss.item() * ACCUMULATION_STEPS 
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Step {batch_idx} | Loss: {loss.item() * ACCUMULATION_STEPS:.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ --- Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {avg_loss:.4f} ---")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"💾 Universal Checkpoint safely backed up to Google Drive for Epoch {epoch+1}.")

    final_deployment_path = os.path.join(DRIVE_SAVE_DIR, 'setu_universal_model_FINAL.pth')
    torch.save(model.state_dict(), final_deployment_path)
    print(f"🎉 Universal Fine-Tuning complete. Final Production weights saved to: {final_deployment_path}")

if __name__ == '__main__':
    train_model()
