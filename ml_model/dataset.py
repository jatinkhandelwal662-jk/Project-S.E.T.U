import os
import json
import numpy as np
import rasterio.features
from shapely.geometry import shape
from PIL import Image
from torch.utils.data import Dataset
import glob

class UniversalRoadDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 1. Find all RGB images (Works for both SpaceNet and OpenSatMap formats)
        self.image_files = sorted(glob.glob(os.path.join(data_dir, '**/*PS-RGB*.tif'), recursive=True))
        
        # 2. Map GeoJSON masks (For Mumbai SpaceNet Data)
        geojson_files = glob.glob(os.path.join(data_dir, '**/*.geojson'), recursive=True)
        self.geojson_map = {os.path.basename(f).split('_chip')[-1].replace('.geojson', ''): f for f in geojson_files}
        
        # 3. Map TIFF masks (For Delhi/Chennai OpenSatMap Data)
        tif_mask_files = glob.glob(os.path.join(data_dir, '**/*GT*.tif'), recursive=True)
        self.tif_mask_map = {os.path.basename(f).split('_chip')[-1].replace('_GT.tif', ''): f for f in tif_mask_files}
        
        print(f"✅ Indexed {len(self.image_files)} images in {os.path.basename(data_dir)}")

    def geojson_to_mask(self, geojson_path, shape_size):
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        
        geoms = [shape(feature['geometry']) for feature in data.get('features', [])]
        if not geoms: 
            return np.zeros(shape_size, dtype=np.uint8)
            
        mask = rasterio.features.rasterize(geoms, out_shape=shape_size, fill=0, default_value=1)
        return mask

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Extract the unique chip ID, handling both naming conventions
        base_name = os.path.basename(img_path)
        if '_chip' in base_name:
            chip_id = base_name.split('_chip')[-1].replace('.tif', '').replace('_PS-RGB', '')
        else:
            chip_id = base_name.replace('.tif', '').replace('_PS-RGB', '')
            
        # Load Image
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size 
        
        # Smart Label Loading: Try GeoJSON first, then TIF, then fallback to blank
        if chip_id in self.geojson_map:
            mask = self.geojson_to_mask(self.geojson_map[chip_id], shape_size=(img_height, img_width))
            mask = (mask * 255).astype(np.uint8)
        elif chip_id in self.tif_mask_map:
            mask_img = Image.open(self.tif_mask_map[chip_id]).convert('L')
            mask = np.array(mask_img)
        else:
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
        mask = Image.fromarray(mask)
        
        # Apply ISRO augmentations
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask is scaled 0-1 for BCE Loss
        return image, mask.float().unsqueeze(0) / 255.0

    def __len__(self):
        return len(self.image_files)
