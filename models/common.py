import torch

def load_checkpoint(ckpt_path):
    """
    ckpt = {
        "epoch": num_epoch,
        "metrics": metrics,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "opt": config,
        "date": str(datetime.now()),
    }
    """
    ckpt = torch.load(ckpt_path)

    return ckpt

if __name__=="__main__":
    pass