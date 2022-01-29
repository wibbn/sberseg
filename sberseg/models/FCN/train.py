from typing import Optional
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from sberseg.models.FCN.model import FCN
from sberseg.data import DataModule
from sberseg.config import config, params

def train(wandb_api_key: Optional[str] = None, gpus: int = 0):
    model = FCN()

    dm = DataModule(
        data_path=params.data.path,
        batch_size=params.data.batch_size
    )
    
    wandb.login(key=(wandb_api_key or config.wandb.api_key))

    wandb_logger = WandbLogger(
        project=config.wandb.project_name,
        save_dir=config.wandb.log_dir
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
    
    return model

if __name__ == '__main__':
    train()