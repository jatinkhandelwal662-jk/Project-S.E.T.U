import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler

from model import AttentionUNet
from topology_loss import ISROCombinedLoss
from dataset import RoadDataset

def train_model():
    # 1. Hyperparameters: Memory-Optimized for Colab T4 GPU
    EFFECTIVE_BATCH_SIZE = 8 
    ACTUAL_BATCH_SIZE = 1
    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // ACTUAL_BATCH_SIZE
    LEARNING_RATE = 2e-4
    EPOCHS = 30  # Reduced to 30 since convergence was proven
    DATA_DIR = "/content/dataset/train"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Colab Cloud Engine Engaged: {device.type.upper()} with AMP enabled")

    if not os.path.exists(DATA_DIR):
        print(f"CRITICAL ERROR: Directory '{DATA_DIR}' not found. Ensure train.zip is extracted.")
        return

    # 2. ISRO Extreme Terrain Pipeline (Resized to 256 to fit Transformer Memory)
    print("Initializing ISRO Extreme Terrain Augmentation Pipeline...")
    isro_transform = A.Compose([
        A.Resize(256, 256), 
        A.RandomFog(fog_coef_range=(0.3, 0.7), p=0.4),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(num_holes_range=(5, 15), hole_height_range=(10, 40), hole_width_range=(10, 40), fill=0, fill_mask=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 3. Data Loaders
    train_dataset = RoadDataset(data_dir=DATA_DIR, transform=isro_transform)
    train_loader = DataLoader(train_dataset, batch_size=ACTUAL_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # 4. Initialize Network
    model = AttentionUNet(img_ch=3, output_ch=1).to(device)
    criterion = ISROCombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = GradScaler()

    # 5. Training Loop with Gradient Accumulation
    print(f"Starting Deep Cloud Training on {len(train_dataset)} images...")
    
    # Secure Save Path (Google Drive)
    drive_save_path = "/content/drive/MyDrive"
    if not os.path.exists(drive_save_path):
        print("WARNING: Google Drive not mounted. Saving locally to /content/")
        drive_save_path = "/content"
        
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad() # Clear gradients initially
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast():
                predictions = model(images)
                # Normalize loss by accumulation steps
                loss = criterion(predictions, masks) / ACCUMULATION_STEPS 

            # Scale loss and backward pass
            scaler.scale(loss).backward()

            # Update weights only after accumulation steps
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # Reset for next accumulation phase

            epoch_loss += loss.item() * ACCUMULATION_STEPS # Log the true loss
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Step {batch_idx} | Loss: {loss.item() * ACCUMULATION_STEPS:.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ --- Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {avg_loss:.4f} ---")
        
        # Save Weights directly to Google Drive so they are never lost
        epoch_save_file = os.path.join(drive_save_path, f"road_unet_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_save_file)

    # Final Save to Drive
    final_save_file = os.path.join(drive_save_path, "road_unet_model_final.pth")
    torch.save(model.state_dict(), final_save_file)
    print(f"🎉 Colab Training complete. Model weights saved securely to {final_save_file}!")

if __name__ == "__main__":
    train_model()
