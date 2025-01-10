config={
    "project": "paperwork",
    "device": "cuda",
    "checkpoint": None,
    "model": "sam_lora",
    "sam_pretrain_weights": None,
    "rank": None,
    "loss_weight": {
        "bce": 0.5,
        "dice": 0.5
    },
    "epochs":10,
    "dataset":"PA",
    "batch_size":2,
    "data_workers": 2,
    "lr":1e-5,
    "fold":0
}