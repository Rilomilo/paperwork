"""
References:
- https://pytorch.org/vision/0.15/transforms.html
- https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html
"""
import os
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.functional import to_tensor, resize

from utils.plot import plot_image

def polygons2masks(num_class, shape, polygons:list, labels:list)-> torch.Tensor:
    """
        Multiple classes support
    """
    mask = np.zeros(shape, dtype=np.uint8)
    
    if polygons:
        instance_masks = []

        for polygon in polygons:
            instance_mask = np.zeros(shape, dtype=np.uint8)
            polygon = np.asarray(polygon, dtype=np.int32)
            polygon=polygon[None] # fillPoly requires shape of [1, N, 2]
            cv2.fillPoly(instance_mask, polygon, color=1)
            instance_masks.append(instance_mask)

        labels=np.asarray(labels)
        instance_masks=np.asarray(instance_masks)
        # draw masks from larger to smaller to prevent smaller masks from being covered by larger masks
        instance_areas=instance_masks.sum(axis=(1,2))
        order=np.argsort(instance_areas)
        labels=labels[order][::-1]
        instance_masks=instance_masks[order][::-1]
        
        for instance_mask, label in zip(instance_masks, labels):
            mask[instance_mask==1]=label

    mask=torch.tensor(mask, dtype=torch.long) # one_hot requires long type
    masks=torch.nn.functional.one_hot(mask, num_class).permute(2,0,1)
        
    return masks

def polygons2instanceMasks(img_size, polygons):
    masks = []
    for polygon in polygons:
        mask = np.zeros(img_size, dtype=np.uint8) # fillPoly doesn't suppot bool type
        polygon = np.asarray(polygon).astype(np.int32) # fillPoly doesn't suppot float type
        polygon=polygon[None] # fillPoly requires shape of [1, N, 2]
        cv2.fillPoly(mask, polygon, color=1)
        masks.append(mask)

    return np.array(masks)

def parse_dataset(meta_file):
    with open(meta_file, 'r') as fp:
        meta=json.load(fp)

    cls2idx={i["name"]: i["id"] for i in meta["classes"]}
    classes=[i["name"] for i in meta["classes"]]
    return cls2idx, classes

class PleomorphicAdenomaDataset(Dataset):
    def __init__(self, path: Path) -> None:
        super().__init__()

        entries=os.listdir(path/"raw")
        json_files=[entry for entry in entries if entry.endswith(".json")]
        json_files.sort()

        self.dir=path/"raw"
        self.json_files=json_files
        self.cls2idx, self.classes=parse_dataset(path/"meta.json")

    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
            return:
                image: (3,1200,1920)  Tensor[torch.float32]
                labels: (n,)          
                masks: (n,1200,1920)  Tensor[torch.uint8]
                image_path: str
        """
        # read image and annotations
        json_file_path=self.dir/self.json_files[idx]
        with open(json_file_path, 'r') as fp:
            data=json.load(fp)

        image_path, annotations=data["imagePath"], data["shapes"]
        
        # process image
        image=Image.open(os.path.join(self.dir, image_path))
        image=to_tensor(image)

        # process annotations
        labels=[]
        polygons=[]
        for annotation in annotations:
            labels.append(self.cls2idx[annotation["label"]])
            polygons.append(annotation["points"])
        
        masks=polygons2masks(len(self.classes), image.shape[-2:], polygons, labels)

        # transform image and masks
        image=resize(image, size=(640, 1024), antialias=False) # 1200x1920 -> 640x1024
        masks=resize(masks, size=(640, 1024), antialias=False) # 1200x1920 -> 640x1024

        return image, masks, image_path
    
    def view(self, idx_ls):
        json_files=[self.json_files[idx] for idx in idx_ls]
        self.json_files=json_files

class TrafficDataset(Dataset):
    def __init__(self, path: Path) -> None:
        super().__init__()

        entries=os.listdir(path/"image")
        entries.sort()

        self.entries=entries
        self.cls2idx, self.classes=parse_dataset(path/"meta.json")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry=self.entries[idx]

        image=Image.open(Path("data/traffic/image")/entry).convert("RGB")
        image=to_tensor(image)

        mask=Image.open(Path("data/traffic/mask")/entry).convert("RGB")
        mask=torch.tensor(np.array(mask), dtype=torch.long) # one_hot requires long type
        mask=torch.max(mask, dim=2)[0] 
        masks=torch.nn.functional.one_hot(mask, len(self.classes)).permute(2,0,1)

        # transform image and masks
        image=resize(image, size=(768, 1024), antialias=False) # 480x640 -> 768x1024
        masks=resize(masks, size=(768, 1024), antialias=False)
        
        return image, masks, entry
    
    def view(self, idx_ls):
        entries=[self.entries[idx] for idx in idx_ls]
        self.entries=entries

class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(root, 'leftImg8bit', split)
        self.labels_dir = os.path.join(root, 'gtFine', split)
        self.image_paths = []
        self.label_paths = []

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            lbl_dir = os.path.join(self.labels_dir, city)
            for file_name in os.listdir(img_dir):
                self.image_paths.append(os.path.join(img_dir, file_name))
                label_name = file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                self.label_paths.append(os.path.join(lbl_dir, label_name))
                
    def __len__(self):
            return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        if self.transform:
            image = self.transform(image)
        
        # Convert label image to tensor and process it as necessary for your task
        # For example, if you want to convert it to a numpy array:
        label = np.array(label, dtype=np.int32)

        return image, label

def get_dataloader(name, fold, batch_size, data_workers, persistent_workers=True):
    """
        dataset: dataset name
            - PA
    """
    if name=="PA":
        path=Path("data/pleomorphic-adenoma/")
        train_dataset=PleomorphicAdenomaDataset(path)
        val_dataset=PleomorphicAdenomaDataset(path)
    elif name=="traffic":
        path=Path("data/traffic/")
        train_dataset=TrafficDataset(path)
        val_dataset=TrafficDataset(path)
    else:
        raise ValueError("Invalid dataset name")

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    splits = list(kfold.split(train_dataset))
    train_idx, val_idx = splits[fold]
    train_dataset.view(train_idx)
    val_dataset.view(val_idx)

    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, num_workers=data_workers, persistent_workers=persistent_workers, drop_last=True, shuffle=True)
    val_dataloader=DataLoader(val_dataset, batch_size=batch_size, num_workers=data_workers, persistent_workers=persistent_workers, drop_last=True, shuffle=True)

    return train_dataset, val_dataset, train_dataloader, val_dataloader

if __name__=="__main__":
    # get_dataloader("traffic", 0, 4, 4)
    dataset=TrafficDataset(Path("data/traffic/"))
    dataset[0]
    pass