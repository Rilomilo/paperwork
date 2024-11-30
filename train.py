from tqdm import tqdm

import torch
import torch.optim as optim

from utils.common import set_seeds
from utils.dataloader import get_dataloader
from utils.metrics import WeightedBinaryDiceLoss, mIoU
from utils.logger import Logger
from config import config
from models import get_model
from val import validate

def train(
        epochs,
        model,
        device,
        loss_weight,
        lr,
        batch_size,
        data_workers,
        fold,
        **kwargs
):
    device=torch.device(device)
    logger=Logger(**config)
    train_dataset, val_dataset, train_dataloader, val_dataloader = get_dataloader("PA", fold, batch_size=batch_size, data_workers=data_workers)
    num_class=len(train_dataset.classes)
    model = get_model(model, output_ch=num_class).to(device)

    loss_fn = WeightedBinaryDiceLoss(loss_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs): # epoch start
        model.train()

        progress=tqdm(train_dataloader)
        logger.begin_epoch(progress, train_dataset.classes)
        for i, (images, masks, _) in enumerate(progress):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss, dice = loss_fn(outputs, masks)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                miou=mIoU(outputs, masks)

            loss=loss.item()
            dice=dice.cpu().detach().numpy()
            miou=miou.cpu().detach().numpy()
            logger.log_step(epoch, epochs, loss, dice, miou)

        logger.log_epoch(phase="train", epoch=epoch)

        is_best_fit, metrics=validate(
            device,
            epoch,
            epochs,
            model,
            val_dataset,
            val_dataloader,
            logger
        )
        logger.log_checkpoint(is_best_fit, epoch, metrics, model, optimizer, config)
    
    logger.finish()

def main():
    params={

    }
    config.update(params)
    print(config)
    train(**config)

if __name__=="__main__":
    set_seeds(0)
    main()
    # 修改学习率调整策略，调整参数位置，增加新数据集