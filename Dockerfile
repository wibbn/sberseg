FROM pytorchlightning/pytorch_lightning

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget \
      python3-tk && \
    rm -rf /var/lib/apt/lists/*

COPY ./sberseg /workdir/sberseg
COPY ./setup.py /workdir

WORKDIR /workdir

RUN pip install -e .

VOLUME [ "/workdir/data" ]

COPY ./config.yaml /workdir
COPY ./params.yaml /workdir
COPY ./scripts /workdir/scripts
