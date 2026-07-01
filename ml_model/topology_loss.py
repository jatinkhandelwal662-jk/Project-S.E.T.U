import torch
import torch.nn as nn
import torch.nn.functional as F

class ISROCombinedLoss(nn.Module):
    """
    Fulfills ISRO Objective: Combined Dice, IoU, and Boundary-Aware Losses.
    Engineered specifically for extreme class imbalance and occlusion-heavy terrain.
    """
    def __init__(self, bce_weight=0.3, dice_weight=0.3, iou_weight=0.2, boundary_weight=0.2):
        super(ISROCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.boundary_weight = boundary_weight
        self.bce = nn.BCEWithLogitsLoss()

    def extract_boundaries(self, masks):
        """
        Uses max pooling as a high-speed morphological dilation operation.
        Subtracting the original mask from the dilated mask leaves only the outer boundary.
        """
        pooled = F.max_pool2d(masks, kernel_size=3, stride=1, padding=1)
        boundary = pooled - masks
        return boundary

    def forward(self, inputs, targets):
        # 1. Base BCE Loss (Calculated on raw logits for stability)
        bce_loss = self.bce(inputs, targets)

        # Convert logits to probabilities for spatial calculations
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # Flatten tensors for mathematical overlap calculations
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        # 2. Dice Loss (Focuses on overall spatial agreement)
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + 1e-6) / (inputs_flat.sum() + targets_flat.sum() + 1e-6)
        dice_loss = 1.0 - dice_score

        # 3. IoU Loss (Jaccard Index - ISRO Requirement)
        union = inputs_flat.sum() + targets_flat.sum() - intersection
        iou_score = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1.0 - iou_score

        # 4. Boundary-Aware Loss (ISRO Requirement)
        # Forces the network to maintain sharp, continuous road edges under tree canopies
        target_boundaries = self.extract_boundaries(targets)
        pred_boundaries = self.extract_boundaries(inputs_sigmoid)
        
        # Mean Squared Error on the boundary pixels
        boundary_loss = F.mse_loss(pred_boundaries, target_boundaries)

        # 5. Final Weighted Combination
        total_loss = (
            (self.bce_weight * bce_loss) + 
            (self.dice_weight * dice_loss) + 
            (self.iou_weight * iou_loss) + 
            (self.boundary_weight * boundary_loss)
        )
        
        return total_loss
