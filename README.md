# Double Perturbation: On the Robustness of Robustness and Counterfactual Bias Evaluation

Robustness and counterfactual bias are usually evaluated on a test dataset. However, are these evaluations robust? In other words, if a model is robust or unbiased on a test set, will the properties still hold under a slightly perturbed test set? In this paper, we propose a "double perturbation" framework to uncover model weaknesses beyond the test dataset. The framework first perturbs the test dataset to construct abundant natural sentences similar to the test data, and then diagnoses the prediction change regarding a single-word substitution. We apply this framework to study two perturbation-based approaches that are used to analyze models' robustness and counterfactual bias. In the experiments, our method attains high success rates (96.0%-99.8%) in finding vulnerable examples and is able to reveal the hidden model bias. More details can be found in our paper:

_Chong Zhang, Jieyu Zhao, Huan Zhang, Kai-Wei Chang, and Cho-Jui Hsieh_, "Double Perturbation: On the Robustness of Robustness and Counterfactual Bias Evaluation", NAACL 2021

<img src="https://raw.githubusercontent.com/chong-z/nlp-second-order-attack/main/img/paper-image-large.jpg" alt="Thumbnail of the paper" width="500px">

## Attack Setup

Verified Environment:
- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- CUDA Version: 11.2

Clone the repo:
```
git clone --recurse-submodules git@github.com:chong-z/nlp-second-order-attack.git
cd nlp-second-order-attack
```

Create a clean environment in Conda or through your favorite virtual environments:
```
conda create --name SOAttack-3.8 python==3.8.3
conda activate SOAttack-3.8
```

Run the setup for PyTorch 1.7 and RTX 30xx GPU.
```
./setup.sh
```

Train a certified classifier through Jia et al. (2019):
```
python libs/jia_certified/src/train.py classification cnn \
  --out-dir model_data/cnn_cert_test -T 60 \
  --full-train-epochs 20 -c 0.8 --random-seed 1234567 \
  -d 100 --pool mean --dropout-prob 0.2 -b 32 \
  --data-cache-dir .cache --save-best-only
```

Attack 10 examples from the SST2 dataset. Note that `cnn_cert_test` is a pre-defined variable in `models/jia_certified.py`.
```
./patched_textattack attack --attack-from-file=biasattack.py:SOBeamAttack \
  --model-from-file=models/jia_certified.py:cnn_cert_test \
  --dataset-from-nlp=glue:sst2:validation --num-examples=10 --shuffle=False
```

## Setup for Xu et al. (2020):

May require torch==1.6. Will add detailed steps later.

## Credits
1. `TextAttack`: https://github.com/QData/TextAttack.
2. `libs/jia_certified`: https://github.com/robinjia/certified-word-sub.
3. `libs/xu_auto_LiRPA`: https://github.com/KaidiXu/auto_LiRPA.
4. See paper for the full list of references.
