import torch.nn as nn
import segmentation_models_pytorch as smp

class ComboLoss(nn.Module):
    """
    Комбинированная функция потерь: Dice Loss + Binary Cross-Entropy.
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        self.bce_loss = nn.BCEWithLogitsLoss()

    def __call__(self, y_pred, y_true):
        # y_pred и y_true имеют размерность (batch_size, 1, H, W)
        dice = self.dice_loss(y_pred, y_true)
        bce = self.bce_loss(y_pred, y_true)
        # Возвращаем взвешенную сумму
        return self.dice_weight * dice + self.bce_weight * bce
