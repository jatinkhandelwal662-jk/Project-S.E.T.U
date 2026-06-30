import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class RoadDataset(Dataset):
    """
    Custom Dataset to load DeepGlobe satellite image patches and masks.
    Handles OpenCV NumPy arrays to integrate seamlessly with Albumentations.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        all_files = os.listdir(data_dir)
        self.images = sorted([f for f in all_files if f.endswith('_sat.jpg')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)
        
        # Load via OpenCV for Albumentations compatibility
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply the ISRO Extreme Terrain transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Extract mask correctly if ToTensorV2 was applied
            if isinstance(mask, torch.Tensor):
                mask = mask.unsqueeze(0).float() / 255.0
        else:
            # Fallback if no transforms are used
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        # Ensure mask is strictly binary
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0

        return image, mask