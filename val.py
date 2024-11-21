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
    for i, (images, masks, image_paths) in enumerate(progress):
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
        # visualize first 2 batch
        if i<=1:
            images=images.cpu().detach().numpy()
            outputs=outputs.cpu().detach().numpy()
            masks=masks.cpu().detach().numpy()
            logger.log_visualization(image_paths, epoch, images, outputs, masks, val_dataset.classes ,dice)

    is_best, metrics=logger.log_epoch(phase="val", epoch=epoch)

    return is_best, metrics


if __name__=="__main__":
    set_seeds(0)