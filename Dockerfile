FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV PATH /opt/conda/bin:$PATH

ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN apt update
RUN apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    espeak \
    && rm -rf /var/lib/apt/lists 

# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /opt/conda \
    && rm miniconda3.sh \
    && /opt/conda/bin/conda install python=3.11 \
    && /opt/conda/bin/conda install pytorch torchvision torchaudio torchtext pytorch-cuda=11.8 -c pytorch -c nvidia \
    && /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda config --set ssl_verify False \
    && pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    && ln -s /opt/conda/bin/pip /usr/local/bin/pip3

# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

WORKDIR /vits2
CMD ["/bin/bash"]