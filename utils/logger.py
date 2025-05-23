import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import wandb
import torch
from torch import nn
from torchvision.transforms.functional import resize

from utils.plot import visualize_label, to_image

wandb.login()

class Logger:
    def __init__(self, project, **params):
        self.run = wandb.init(
            # Set the project where this run will be logged
            project=project,
            # Track hyperparameters and run metadata
            config=params,
        )

        time=datetime.now().strftime("%Y-%m-%d-%H%M%S")
        log_dir=Path(f"output/{project}{time}")
        log_dir.mkdir()

        self.log_dir=log_dir
        self.max_validation_mixed_metric=0
        self.viz_table = wandb.Table(columns=["name", "epoch", "pred", "label", "dice"])

    def log_metrics(self, metrics, step=None, commit=True):
        self.run.log(metrics, step=step, commit=commit)

    def begin_epoch(self, epoch_progress, classes):
        self.progress=epoch_progress
        self.classes=classes[1:] # ignore background class

        self.step_loss_metrics=[] # [step_length]
        self.step_dice_metrics=[] # [step_length, num_class]
        self.step_miou_metrics=[] # [step_length, num_class]

        print(("\n" + "%12s" * 4)% ("Epoch", "Loss", "Dice", "mIoU"))

    def log_step(self, epoch, epochs, loss:float, dice:np.ndarray, miou:np.ndarray, phase, lr=None):
        self.step_loss_metrics.append(loss)
        self.step_dice_metrics.extend(dice)
        self.step_miou_metrics.extend(miou)

        self.progress.set_description(
            ("%12s" * 1 + "%12.4g" * 3) % (
                f"{epoch+1}/{epochs}",
                loss,
                dice.mean(),
                miou.mean()
            )
        )

        if phase=="train":
            metrics={
                "loss": loss,
                "lr": lr
            }
            self.log_metrics(metrics, commit=True)

    def log_epoch(self, phase, epoch):
        """
            Params:
                phase: ["train", "val"]
            Returns:
                - if it's best fitness
                - metrics
        """
        loss = np.array(self.step_loss_metrics).mean()
        dice = np.array(self.step_dice_metrics).mean(axis=0)
        miou = np.array(self.step_miou_metrics).mean(axis=0)

        dice_metric={f"{class_name}": dice_score for class_name,dice_score in zip(self.classes,dice)}
        miou_metric={f"{class_name}": miou_score for class_name,miou_score in zip(self.classes,miou)}
        dice_metric.update({"avg": dice.mean()})
        miou_metric.update({"avg": miou.mean()})
        mixed_metric = 0.5 * dice_metric["avg"] + 0.5 * miou_metric["avg"]

        metrics={
            "epoch": epoch,
            f"{phase}/loss": loss,
            f"{phase}/dice": dice_metric,
            f"{phase}/miou": miou_metric
        }
        print(metrics)
        self.log_metrics(metrics, commit=False) # commit in training log step

        is_best_fit=mixed_metric>self.max_validation_mixed_metric

        if phase=="val" and is_best_fit:
            self.max_validation_mixed_metric = mixed_metric
            return True, metrics
        else:
            return False, metrics
        
    def log_checkpoint(self, is_best_fit, num_epoch, step, metrics, model:nn.Module, optimizer,warmup_scheduler, decay_scheduler, config):
        ckpt = {
            "epoch": num_epoch,
            "step": step,
            "metrics": metrics,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "warmup_scheduler": warmup_scheduler.state_dict(),
            "decay_scheduler": decay_scheduler.state_dict(),
            "opt": config,
            "date": str(datetime.now()),
        }

        torch.save(ckpt, self.log_dir / "latest.pt")
        if is_best_fit:
            shutil.copy(self.log_dir / "latest.pt", self.log_dir / "best.pt")
            print("Best model saved")

    def log_visualization(self, names, epoch, images, preds, labels, classes, dice, mode="wandb"):
        # size=224 # resize images to improve loading speed
        # images = resize(images, size=size, antialias=False)
        # preds = resize(preds, size=size, antialias=False)
        # labels = resize(labels, size=size, antialias=False)
        images=images.cpu().detach().numpy()
        preds=preds.cpu().detach().numpy()
        labels=labels.cpu().detach().numpy()
        dice = dice.round(4) # keep 4 decimal places only
        
        for name, image, pred, label, dice_score in zip(names, images, preds, labels, dice):
            image=to_image(image)
            if mode=="wandb":
                label=visualize_label(image, label, classes)
                pred=visualize_label(image, pred, classes)
                pred = wandb.Image(pred)
                label = wandb.Image(label)
                dice_score=dict(zip(classes[1:], list(dice_score)))

                self.viz_table.add_data(name, epoch, pred, label, dice_score)
            elif mode=="local":
                visualize_label(image, label, classes, output_path=self.log_dir/f"{name}_label.jpg")
                visualize_label(image, pred, classes, output_path=self.log_dir/f"{name}_pred.jpg")

    def finish(self):
        self.run.log({"Visualization": self.viz_table})

if __name__=="__main__":
    pass