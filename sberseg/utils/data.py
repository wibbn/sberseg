import torch
import wandb
import numpy as np
import cv2

import torchvision.transforms as transforms

TRANSFORM_MEAN = [0.35675976, 0.37380189, 0.3764753]
TRANSFORM_STD = [0.32064945, 0.32098866, 0.32325324]

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = TRANSFORM_MEAN, std = TRANSFORM_STD)
])

def unnormalize(tensor: torch.Tensor, mean: float, std: float):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def wb_image(img, mask):

    labels = {
        0: 'Building',
        1: 'Road',
        2: 'Static car',
        3: 'Tree',
        4: 'Low vegetation',
        5: 'Human',
        6: 'Moving car',
        7: 'Background clutter',
    }

    img = unnormalize(img, mean=TRANSFORM_MEAN, std=TRANSFORM_STD)
    img = img.cpu().transpose(0, 2).transpose(0, 1).numpy()

    mask = np.argmax(mask.cpu().numpy(), axis=0)
    

    return wandb.Image(img, masks={
        "prediction" : {
            "mask_data" : mask, "class_labels" : labels
        }
    })


def get_single_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (960, 540), )
    img = data_transform(img)

    return img[None, :]