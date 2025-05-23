import numpy as np
from PIL import Image
import imgviz

import torch

def to_image(data):
    """
    Convert Tensor to image
    Returns: np.ndarray
    """
    if isinstance(data, Image.Image):
        pass
    elif isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    else:
        data = np.array(data)

    if isinstance(data, np.ndarray):
        # strip extra dims
        if len(data.shape)>3:
            data=data.reshape(data.shape[-3:])

        # move RGB channel to the latest
        if len(data.shape)==3:
            channel_index=np.array(data.shape).argmin()
            if channel_index==0:
                data = data.transpose(1, 2, 0)

        # unnormalize
        if data.max()<=1:
            data*=255

        # PIL.Image needs uint8 type
        data = data.astype(np.uint8)

    return data

def plot_image(data, path="plot.jpg"):
    data=to_image(data)
    data = Image.fromarray(data)
    data.save(path)

def visualize_label(image, masks, label_names, output_path=None)-> np.ndarray:
    image=to_image(image)

    num_class=len(label_names)
    label_names=label_names+["overlap"]
    
    masks=masks>0.5
    overlap_mask=masks.sum(axis=0)>1
    composed_mask=np.zeros(shape=masks.shape[-2:], dtype=np.uint8)

    for idx, layer in enumerate(masks):
        composed_mask[layer]=idx
    composed_mask[overlap_mask]=num_class

    viz = imgviz.label2rgb(composed_mask, image, label_names=label_names, font_size=12)
    if output_path:
        Image.fromarray(viz).save(output_path)

    return viz

if __name__=="__main__":
    data=torch.ones((1,224,224,3))
    plot_image(data)