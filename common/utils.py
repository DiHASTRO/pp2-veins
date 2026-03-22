import torch
import numpy as np

def evaluate_metrics(model, loader, device, num_classes):
    """Вычисляет IoU и Dice для каждой маски, возвращает средние."""
    model.eval()
    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            pred_masks = model.predict(images)  # (B, H, W)

            for cls in range(num_classes):
                pred_cls = (pred_masks == cls).float()
                true_cls = (masks == cls).float()
                intersection = (pred_cls * true_cls).sum((1, 2))
                union = pred_cls.sum((1, 2)) + true_cls.sum((1, 2)) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                iou_scores.extend(iou.cpu().numpy())

            smooth = 1e-6
            intersection = (pred_masks * masks).sum((1, 2)).float()
            dice = (2. * intersection + smooth) / (pred_masks.sum((1, 2)).float() + masks.sum((1, 2)).float() + smooth)
            dice_scores.extend(dice.cpu().numpy())

    return {
        'mean_iou': np.mean(iou_scores),
        'mean_dice': np.mean(dice_scores)
    }
