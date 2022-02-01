from typing import Optional
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from sberseg.data import UAVidDataModule
from sberseg.config import config, params

def test(model: pl.LightningModule, state_dict_path: str, wandb_api_key: Optional[str] = None, gpus: int = 0):
    dm = UAVidDataModule(
        data_path=params.data.path,
        batch_size=params.data.batch_size
    )
    
    wandb_api_key = wandb_api_key or config.wandb.api_key
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    wandb_logger = WandbLogger(
        project=config.wandb.project_name,
        save_dir=config.wandb.log_dir,
        offline=(not wandb_api_key)
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=2,
        max_epochs = params.learning.epochs,
        gpus=gpus,
    )

    model.load_state_dict(torch.load(state_dict_path))

    trainer.test(model, dm)
