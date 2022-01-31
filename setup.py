import setuptools

setuptools.setup(
    name='sberseg',
    packages=setuptools.find_packages(),
    install_requires=[
        'wandb',
        'pytorch-lightning',
        'numpy',
        'opencv-python',
        'python-box',
        'click'
    ]
)