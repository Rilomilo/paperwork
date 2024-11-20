from tqdm import tqdm

import torch

from utils.common import set_seeds
from utils.metrics import WeightedBinaryDiceLoss, dice_coefficient, mIoU
from utils.logger import Logger

def validate(
        device,
        epoch,
        epochs,
        model,
        loss_weight,
        val_dataset,
        val_dataloader,
        logger: Logger
):
    device=torch.device(device)

    loss_fn = WeightedBinaryDiceLoss(loss_weight, softmax=True)

    model=model.to(device)
    model.eval()

    progress=tqdm(val_dataloader)
    logger.begin_epoch(progress, val_dataset.classes)
    for images, masks, _ in progress:
        images, masks = images.to(device), masks.to(device)

        with torch.no_grad():
            outputs = model(images)
            loss, _ = loss_fn(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            miou = mIoU(outputs, masks)
            
        loss=loss.item()
        dice=dice.cpu().detach().numpy()
        miou=miou.cpu().detach().numpy()
        logger.log_step(epoch, epochs, loss, dice, miou)

    is_best, metrics=logger.log_epoch(phase="val", epoch=epoch)

    return is_best, metrics


if __name__=="__main__":
    set_seeds(0)