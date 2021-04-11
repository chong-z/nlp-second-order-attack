#!/bin/bash

./quick_setup.sh

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

# Setup word_embeddings for training GN models with TextAttack
mkdir -p word_embeddings
cd word_embeddings
wget -O GN-GloVe-L1-0.8-0.8.txt.zip https://www.dropbox.com/s/o42ihpdr27ha111/GN-GloVe-L1-0.8-0.8.txt.zip
unzip GN-GloVe-L1-0.8-0.8.txt.zip
mv vectors300.txt gn_glove.zhao2018.wikidump.300d.txt
wget -O GloVe.zip https://www.dropbox.com/s/hjghi71sbvay0wn/GloVe.zip
mv vectors.txt glove.zhao2018.wikidump.300d.txt
cd ..
