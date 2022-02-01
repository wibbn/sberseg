import torch
import pytorch_lightning as pl
from sberseg.utils.data import get_blended_output, get_single_image, save_image
from sberseg.config import config

def inference(model: pl.LightningModule, state_dict_path: str, image_path: str):

    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    img = get_single_image(image_path)
    with torch.no_grad():
        out = model(img)[0]

    save_path = save_image(get_blended_output(img[0], out), config.media.output_dir)

    print(f'You can find the markup result image here {save_path} . Enjoy it!')