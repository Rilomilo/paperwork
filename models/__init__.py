import segmentation_models_pytorch as smp
from torch import nn
import torchinfo

from .unet import U_Net
from .common import load_checkpoint
from .segment_anything import build_sam_vit_b
from .segment_anything.sam_lora import LoRA_Sam

def get_model(name, output_ch, weights=None, rank=None):
    if name=="U_Net":
        model=U_Net(img_ch=3, output_ch=output_ch)
    elif name=="u-resnet34":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=output_ch
        )
        model.segmentation_head.add_module("softmax", nn.Softmax(dim=1))
    elif name=="sam":
        model = build_sam_vit_b(output_ch, "weights/sam_vit_b_01ec64.pth")
    elif name=="sam_lora":
        model = build_sam_vit_b(output_ch, "weights/sam_vit_b_01ec64.pth")
        sam_lora=LoRA_Sam(model, r=rank)
    else:
        raise ValueError(f"Model {name} not found")
    
    if weights:
        model.load_state_dict(weights)

    torchinfo.summary(model, depth=10)
    
    return model