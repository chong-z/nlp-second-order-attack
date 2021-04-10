#!/bin/bash

mkdir -p .cache
mkdir -p data
mkdir -p model_data

# RTX 30xx requires torch>=1.7.1. Torch 1.6 should also work if you have a compatible GPU.
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining requirements.
pip install -r requirements.txt

# lm-scorer has weird dependency on transformers<3.0.0,>=2.9.0 and python<3.8.
pip install --no-deps --ignore-requires-python lm-scorer==0.4.2

# Setup Jia et al. (2019).
cd libs/jia_certified
./download_deps.sh
cd ../..

# Setup Xu et al. (2020).
ln -s ../../../../model_data libs/xu_auto_LiRPA/examples/language/model_data
ln -s ../../../../data libs/xu_auto_LiRPA/examples/language/data
cd libs/xu_auto_LiRPA/examples/language
wget http://download.huan-zhang.com/datasets/language/data_language.tar.gz
tar xvfk data_language.tar.gz
cd ../../../..
