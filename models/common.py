import torch

def load_checkpoint(ckpt_path):
    """
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
    """
    ckpt = torch.load(ckpt_path)

    return ckpt

if __name__=="__main__":
    pass