______________________________________________________________________

<div align="center">

# VITS2

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Unofficial Lightning implementation of **VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design** (Kong et al. *Interspeech 2023*) [[Arxiv]](https://arxiv.org/abs/2307.16430)

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/Normalist-K/lightning-vits2
cd lightning-vits2

# [OPTIONAL] create conda environment
conda create -n myenv python=3.11
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
# You may need to install espeak first: apt-get install espeak
```

#### Conda

```bash
# clone project
git clone https://github.com/Normalist-K/lightning-vits2
cd lightning-vits2

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## Prerequisites
#### Download datasets (Available multi-speaker only for now)
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    2. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
#### Build Monotonic Alignment Search
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace
```
#### Run preprocessing if you use your own datasets
```
# python src/utils/preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python src/utils/preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```
## How to run

Train model with default configuration

```bash
python src/train.py trainer=gpu model=vits2_multi data=vctk logger=tensorboard
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

Dry run for debug

```bash
python src/train.py debug=defualt data=vctk_dev 
python src/train.py debug=limit # use only small portion of the data
```

Code follows the structure of the lightning-hydra-template and it supports a lot of amazing things.
Check here, [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

## TODOs, features and notes
- [ ] Implement single speaker training code
- [ ] Implement inference code
- [ ] Updates pretrained model


## Acknowledgements
This repository is highly inspired by PyTorch [VITS2](https://github.com/p0p4k/vits2_pytorch) repository.
