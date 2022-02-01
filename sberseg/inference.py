from typing import Optional
import torch
import pytorch_lightning as pl
from sberseg.utils.data import get_single_image

def inference(model: pl.LightningModule, state_dict_path: str, image_path: str):

    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    img = get_single_image(image_path)
    with torch.no_grad():
        out = model(img)[0]

    print(out.shape)
