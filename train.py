from tqdm import tqdm

import torch
import torch.optim as optim

from utils.common import set_seeds
from utils.dataloader import get_dataloader
from utils.metrics import WeightedBinaryDiceLoss, mIoU, dice_coefficient
from utils.logger import Logger
from utils.cli_parser import parse_args
from config import config
from models import get_model, load_checkpoint
from val import validate

def train(
        epochs,
        model,
        device,
        loss_weight,
        lr,
        dataset,
        batch_size,
        data_workers,
        fold,
        **kwargs
):
    device=torch.device(device)
    logger=Logger(**config)
    train_dataset, val_dataset, train_dataloader, val_dataloader = get_dataloader(dataset, fold, batch_size=batch_size, data_workers=data_workers)
    num_class=len(train_dataset.classes)
    model = get_model(model, output_ch=num_class, rank=config["rank"], sam_pretrain_weights=config["sam_pretrain_weights"]).to(device)

    loss_fn = WeightedBinaryDiceLoss(loss_weight)

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.1)
    warmup_steps=250
    warmup_scheduler=torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=lr, 
        total_iters=warmup_steps
    )
    decay_scheduler=optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        factor=0.3, 
        patience=0, 
        threshold=1e-3, 
        threshold_mode="rel", 
        verbose=True
    )
    step=0

    if config["checkpoint"]:
        ckpt = load_checkpoint(config["checkpoint"])
        epoch, step, model_state, optimizer_state, warmup_scheduler_state, decay_scheduler_state, ckpt_config = ckpt["epoch"], ckpt["step"], ckpt["model"], ckpt["optimizer"], ckpt["warmup_scheduler"], ckpt["decay_scheduler"], ckpt["opt"]
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        warmup_scheduler.load_state_dict(warmup_scheduler_state)
        decay_scheduler.load_state_dict(decay_scheduler_state)
        epochs = ckpt_config["epochs"]

    for epoch in range(epochs): # epoch start
        model.train()

        progress=tqdm(train_dataloader)
        logger.begin_epoch(progress, train_dataset.classes)
        for i, (images, masks, names) in enumerate(progress):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                # turn scores to one hot
                outputs = outputs.argmax(axis=1)
                outputs = torch.nn.functional.one_hot(outputs, len(val_dataset.classes)).permute(0,3,1,2)

                dice = dice_coefficient(outputs, masks)
                miou = mIoU(outputs, masks)

            loss=loss.item()
            dice=dice.cpu().detach().numpy()
            miou=miou.cpu().detach().numpy()
            logger.log_step(epoch, epochs, loss, dice, miou, phase="train", lr=optimizer.state_dict()['param_groups'][0]['lr'])
            warmup_scheduler.step()
            step+=1

        _, train_metrics = logger.log_epoch(phase="train", epoch=epoch)

        is_best_fit, val_metrics=validate(
            device,
            epoch,
            epochs,
            model,
            val_dataset,
            val_dataloader,
            logger
        )
        logger.log_checkpoint(is_best_fit, epoch, step, val_metrics, model, optimizer, warmup_scheduler, decay_scheduler, config)

        # end training according to lr
        decay_scheduler.step(train_metrics["train/loss"])
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        if step>warmup_steps and lr<1e-7:
            break
    
    logger.finish()

def main():
    parse_args(config)
    print(config)
    train(**config)

if __name__=="__main__":
    set_seeds(0)
    main()
    