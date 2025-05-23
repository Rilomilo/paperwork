from tqdm import tqdm

import torch

from config import config
from utils.common import set_seeds
from utils.metrics import WeightedBinaryDiceLoss, dice_coefficient, mIoU
from utils.logger import Logger
from utils.dataloader import get_dataloader
from utils.cli_parser import parse_args
from models import load_checkpoint, get_model

def validate(
        device,
        epoch,
        epochs,
        model,
        val_dataset,
        val_dataloader,
        logger: Logger
):
    device=torch.device(device)

    loss_fn = WeightedBinaryDiceLoss(config["loss_weight"])

    model=model.to(device)
    model.eval()

    progress=tqdm(val_dataloader)
    logger.begin_epoch(progress, val_dataset.classes)
    for i, (images, masks, image_paths) in enumerate(progress):
        images, masks = images.to(device), masks.to(device)

        with torch.no_grad():
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            # turn scores to one hot
            outputs = outputs.argmax(axis=1)
            outputs = torch.nn.functional.one_hot(outputs, len(val_dataset.classes)).permute(0,3,1,2)

            dice = dice_coefficient(outputs, masks)
            miou = mIoU(outputs, masks)
            
        loss=loss.item()
        dice=dice.cpu().detach().numpy()
        miou=miou.cpu().detach().numpy()

        logger.log_step(epoch, epochs, loss, dice, miou, phase="val")
        # visualize first 2 batch
        if i<=1:
            logger.log_visualization(image_paths, epoch, images, outputs, masks, val_dataset.classes, dice, "wandb")

    is_best, metrics=logger.log_epoch(phase="val", epoch=epoch)

    return is_best, metrics

def main():
    parse_args(config)
    print(config)

    device, model, rank, dataset, batch_size, data_workers, fold, checkpoint=config["device"], config["model"], config["rank"], config["dataset"], config["batch_size"], config["data_workers"], config["fold"], config["checkpoint"]

    train_dataset, val_dataset, train_dataloader, val_dataloader = get_dataloader(dataset, fold, batch_size=batch_size, data_workers=data_workers)
    
    ckpt = load_checkpoint(checkpoint)
    epoch, weights, ckpt_config = ckpt["epoch"], ckpt["model"], ckpt["opt"]
    epochs=ckpt_config["epochs"]

    model = get_model(config["model"], output_ch=len(val_dataset.classes), weights=weights, rank=rank)

    logger = Logger(**config)

    validate(
        device,
        epoch,
        epochs,
        model,
        val_dataset,
        val_dataloader,
        logger
    )

    logger.finish()

if __name__=="__main__":
    set_seeds(0)
    main()