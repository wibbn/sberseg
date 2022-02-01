# Sematic Segmentation models for UAVid Dataset

## Data

1. You can download data from this link https://www.kaggle.com/dasmehdixtr/uavid-v1. In order to do this, you need to log in to Kaggle.
2. To use the data you need to unpack it into the `data` directory. So that the file structure would be as follows. Or you can change the corresponding line in the file `config.yaml`

```
\data
    \uavid_test
    ...
    \uavid_val
```

## Configuration

+ The parameters of the models and the learning process are set by files `config.yaml` and `params.yaml`. 
+ Parameters responsible for the work script are passed as arguments to the executable file

## Build and run

You can use docker or your native environment to run this code.

For run this project with docker you need to run some commands in your terminal. Run docker-compose with executable command at the end. For example
```bash
docker-compose run sberseg \
python sberseg/main.py --model FCN --stage inference --checkpoint checkpoints/fcn.pth --image data/uavid_test/seq21/Images/000000.png
```
Here after the command `python sberseg/main.py` you can pass the required arguments.
