from .unet import U_Net
from .common import load_checkpoint

def get_model(name, output_ch, weights=None):
    if name=="U_Net":
        model=U_Net(img_ch=3, output_ch=output_ch)
    else:
        raise ValueError(f"Model {name} not found")
    
    if weights:
        model.load_state_dict(weights)
    
    return model