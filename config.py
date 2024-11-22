config={
    "project": "paperwork",
    "device": "cuda",
    "model": "U_Net",
    "loss_weight": {
        "bce": 0.5,
        "dice": 0.5
    },
    "epochs":10,
    "batch_size":4,
    "data_workers": 2,
    "lr":1e-4,
    "fold":0
}