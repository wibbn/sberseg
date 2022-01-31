from typing import Optional
import click
import pytorch_lightning as pl
import wandb
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
    
    return model

@click.command()
@click.option('-m', '--model', 'model_name', type=click.Choice(['FCN', 'FastFCN'], case_sensitive=False))
def main(model_name: str):
    from sberseg.models import FCN, FastFCN

    model=None
    if model_name == 'FCN':
        model = FCN(
            num_classes=params.data.num_classes,
            learning_rate=params.model.FCN.learning_rate
        )
    elif model_name == 'FastFCN':
        model = FastFCN(
            num_classes=params.data.num_classes,
            learning_rate=params.model.FastFCN.learning_rate
        )

    train(model)
    

if __name__ == '__main__':

    main()