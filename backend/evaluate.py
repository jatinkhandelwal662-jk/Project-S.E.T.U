import torch
from model import TransUNet # Ensure this matches your model class
from torch.utils.data import DataLoader

def evaluate_model(model, test_loader, device):
    model.eval()
    total_iou = 0
    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            
            # Binary threshold for road detection
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            # Calculate IoU
            intersection = (preds * masks).sum()
            union = (preds + masks).sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            total_iou += iou.item()
            
    print(f"Final Model IoU Score: {total_iou / len(test_loader):.4f}")

# Usage
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# evaluate_model(model, test_loader, device)