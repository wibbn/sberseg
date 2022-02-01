from typing import Optional
import click

from sberseg.config import params
from sberseg.inference import inference
from sberseg.train import train
from sberseg.test import test

@click.command()
@click.option('-m', '--model', 'model_name', type=click.Choice(['FCN', 'FastFCN'], case_sensitive=False), required=True)
@click.option('-s', '--stage', type=click.Choice(['train', 'test', 'inference'], case_sensitive=False), required=True)
@click.option('-c', '--checkpoint', 'model_checkpoint', type='str')
@click.option('-i', '--image', 'image_path', type='str')
def main(model_name: str, stage: str, model_checkpint: Optional[str] = None, image_path: Optional[str] = None):
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

    if stage == 'train':
        train(model)
    elif stage == 'test':
        test(model, state_dict_path=model_checkpint)
    elif stage == 'inference':
        inference(model, state_dict_path=model_checkpint, image_path=image_path)
    

if __name__ == '__main__':
    main()
