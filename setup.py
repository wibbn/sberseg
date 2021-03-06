import setuptools

setuptools.setup(
    name='sberseg',
    packages=setuptools.find_packages(),
    install_requires=[
        'wandb==0.12.10',
        'pytorch-lightning==1.5.3',
        'opencv-python==4.5.5.62',
        'python-box==5.4.1',
        'click==8.0.3'
    ]
)