import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler

from model import AttentionUNet
from topology_loss import DiceBCELoss
from dataset import RoadDataset

def train_model():
    # 1. Hyperparameters Optimized for Local GPU
    BATCH_SIZE = 8 # INCREASED: AMP uses less memory, so we can process double the images at once!
    LEARNING_RATE = 1e-4
    EPOCHS = 5 # REDUCED: 5 epochs is plenty for a hackathon demo
    DATA_DIR = "./train" 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Accelerated on: {device.type.upper()} with AMP enabled")

    if not os.path.exists(DATA_DIR):
        print(f"CRITICAL ERROR: Directory '{DATA_DIR}' not found. Please ensure dataset is unzipped.")
        return

    # 3. ISRO Extreme Terrain Pipeline (Adversarial Occlusion Training)
    print("Initializing ISRO Extreme Terrain Augmentation Pipeline...")
    isro_transform = A.Compose([
        A.Resize(512, 512),
        A.RandomFog(fog_coef_range=(0.3, 0.7), p=0.4),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(num_holes_range=(5, 15), hole_height_range=(10, 40), hole_width_range=(10, 40), fill=0, fill_mask=0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 4. Data Loaders (TURBOCHARGED)
    train_dataset = RoadDataset(data_dir=DATA_DIR, transform=isro_transform)
    # Added pin_memory=True to build a fast-lane direct to the GPU VRAM
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    # 5. Initialize Network
    model = AttentionUNet(img_ch=3, output_ch=1).to(device)
    criterion = DiceBCELoss(bce_weight=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # INITIALIZE THE AMP SCALER
    scaler = GradScaler()

    # 6. Training Loop
    print(f"Starting Turbo Model Training on {len(train_dataset)} images...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True) # non_blocking speeds up transfer
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            # THE NITROUS BOOST: Run forward pass in 16-bit Mixed Precision
            with autocast():
                predictions = model(images)
                loss = criterion(predictions, masks)

            # Scale loss and backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            
            # Print an update every 20 batches since it will be moving much faster
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        print(f"✅ --- Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {epoch_loss / len(train_loader):.4f} ---")
        
        # Save Weights per epoch
        torch.save(model.state_dict(), "road_unet_model.pth")

    print("🎉 Turbo Training complete. Model weights saved to 'road_unet_model.pth'")

# 7. THE IGNITION SWITCH (Required to run the script)
if __name__ == "__main__":
    train_model()