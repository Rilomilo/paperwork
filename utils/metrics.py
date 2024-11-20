import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, target, weight=None):
        """
            preds & target: [N, n_classes, H, W]
            dice = 2 * intersect/(preds_area + target_area)
        """
        smooth = 1e-5
        intersect = torch.sum(preds * target, dim=(-1, -2))
        target_area = torch.sum(target, dim=(-1, -2))
        preds_area = torch.sum(preds , dim=(-1, -2))
        dice = (2 * intersect + smooth) / (preds_area + target_area + smooth) # [N, n_classes]
        loss = (1 - dice).mean()

        return loss, dice

class WeightedBinaryDiceLoss(nn.Module):
    def __init__(self, loss_weight, softmax):
        super(WeightedBinaryDiceLoss, self).__init__()
        self.dice_fn = DiceLoss()
        self.bce_fn= BCEWithLogitsLoss()
        self.weight= loss_weight
        self.softmax = softmax
    
    def forward(self, prediction, mask:torch.Tensor):
        """
            prediction: torch.float32[batch_size, n_classes, H, W]
            mask: 
        """
        if self.softmax:
            prediction = torch.softmax(prediction, dim=1)
        else:
            assert prediction.max() <= 1 and prediction.min()>=0, "prediction scores should be between 0 and 1"

        # bce_fn needs preds and target to be the same type
        if mask.dtype==torch.uint8:
            mask = mask.float() # torch.float32

        dice_loss, dice = self.dice_fn(prediction, mask)
        bce_loss = self.bce_fn(prediction, mask)
        loss= self.weight["bce"]*bce_loss + self.weight["dice"]*dice_loss

        return loss, dice

class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, preds, target, weight=None):
        """
            preds & target: [N, n_classes, H, W]
            dice = 2 * intersect/(preds_area + target_area)
        """
        smooth = 1e-5
        intersect = torch.sum(preds * target, dim=(-1, -2))
        target_area = torch.sum(target, dim=(-1, -2))
        preds_area = torch.sum(preds , dim=(-1, -2))
        dice = (2 * intersect + smooth) / (preds_area + target_area + smooth) # [N, n_classes]
        loss = (1 - dice).mean()

        return loss, dice

# def dice_coefficient(pred, target, threshold=0.5):
#     pred = (pred > threshold).astype(int)  # 将Softmax输出转为二值
#     intersection = np.sum(pred * target)
#     return (2. * intersection) / (np.sum(pred) + np.sum(target))

if __name__=="__main__":
    dice=Dice()
    