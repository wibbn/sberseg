import os
import torch
import wandb
import numpy as np
import cv2
import uuid

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

def data_untransform(tensor: torch.Tensor):
    return unnormalize(tensor, TRANSFORM_MEAN, TRANSFORM_STD).cpu().transpose(0, 2).transpose(0, 1).numpy() * 255

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

    img = data_untransform(img)
    mask = np.argmax(mask.cpu().numpy(), axis=0)

    return wandb.Image(img, masks={
        "prediction" : {
            "mask_data" : mask, "class_labels" : labels
        }
    })

def get_single_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (960, 540))
    img = data_transform(img)

    return img[None, :]

def get_colors_from_class(num):
    return [128 if num & i else 0 for i in [4, 2, 1]]

def image_from_mask(mask):
    mask = np.argmax(mask.cpu().numpy(), axis=0)
    colored_mask = np.array([[get_colors_from_class(num) for num in x] for x in mask])

    return colored_mask

def get_blended_output(img, mask, alpha: float = 0.55):
    img = data_untransform(img)
    mask = image_from_mask(mask)

    out = img * alpha + mask * (1-alpha)

    return out

def save_image(img, path):
    filename = f'{uuid.uuid4().hex}.png'
    save_path = os.path.join(path, filename)
    cv2.imwrite(save_path, img)

    return save_path