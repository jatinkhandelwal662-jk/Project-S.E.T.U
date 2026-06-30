import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    """
    Combines Binary Cross-Entropy and Dice Loss.
    Crucial for road extraction because roads represent a tiny fraction 
    of the overall image (extreme class imbalance).
    """
    def __init__(self, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        # UPGRADED: BCEWithLogitsLoss handles models that do not have a final Sigmoid layer
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # Calculate standard BCE with logits directly
        bce_loss = self.bce(inputs.view(-1), targets.view(-1))
        
        # Manually apply Sigmoid for the Dice calculation
        inputs_sigmoid = torch.sigmoid(inputs).view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Dice Loss
        intersection = (inputs_sigmoid * targets_flat).sum()
        dice_score = (2. * intersection + 1e-6) / (inputs_sigmoid.sum() + targets_flat.sum() + 1e-6)
        dice_loss = 1.0 - dice_score
        
        # Weighted combination
        return (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)