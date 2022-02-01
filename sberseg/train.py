from typing import Optional
import uuid
import pytorch_lightning as pl
import wandb
import torch
from pytorch_lightning.loggers import WandbLogger

from sberseg.data import UAVidDataModule
from sberseg.config import config, params

def train(model: pl.LightningModule, wandb_api_key: Optional[str] = None, gpus: int = 0):
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

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath = 'checkpoints/SegModel/',
        save_top_k=3,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        verbose = True, 
        monitor = 'val/loss',
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=2,
        max_epochs = params.learning.epochs,
        gpus=gpus,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, dm)
    
    modelname = f'{uuid.uuid4().hex}.png'
    torch.save(model.state_dict(), f'{config.checkpoints.path}/model-{modelname}.pth')