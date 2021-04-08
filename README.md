# Double Perturbation: On the Robustness of Robustness and Counterfactual Bias Evaluation

Robustness and counterfactual bias are usually evaluated on a test dataset. However, are these evaluations robust? In other words, if a model is robust or unbiased on a test set, will the properties still hold under a slightly perturbed test set? In this paper, we propose a "double perturbation" framework to uncover model weaknesses beyond the test dataset. The framework first perturbs the test dataset to construct abundant natural sentences similar to the test data, and then diagnoses the prediction change regarding a single-word substitution. We apply this framework to study two perturbation-based approaches that are used to analyze models' robustness and counterfactual bias. In the experiments, our method attains high success rates (96.0%-99.8%) in finding vulnerable examples and is able to reveal the hidden model bias. More details can be found in our paper:

_Chong Zhang, Jieyu Zhao, Huan Zhang, Kai-Wei Chang, and Cho-Jui Hsieh_, "Double Perturbation: On the Robustness of Robustness and Counterfactual Bias Evaluation", NAACL 2021

<img src="https://raw.githubusercontent.com/chong-z/nlp-second-order-attack/main/img/paper-image-large.jpg" alt="Thumbnail of the paper" width="500px">

## Setup

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

## Run Attacks

### Certified BoW, CNN, and LSTM (Jia et al., 2019)

Train a certified CNN model:
```
python libs/jia_certified/src/train.py classification cnn \
  --out-dir model_data/cnn_cert_test -T 60 \
  --full-train-epochs 20 -c 0.8 --random-seed 1234567 \
  -d 100 --pool mean --dropout-prob 0.2 -b 32 \
  --data-cache-dir .cache --save-best-only
```

Attack 10 examples from the SST2 dataset. Note that `cnn_cert_test` is a pre-defined variable in `models/jia_certified.py`, and you need to modify the file if you are using a different `--out-dir`.
```
./patched_textattack attack --attack-from-file=biasattack.py:SOBeamAttack \
  --dataset-from-nlp=glue:sst2:validation --num-examples=10 --shuffle=False \
  --model=models/jia_certified.py:cnn_cert_test
```

### Certified Transformers (Xu et al. 2020)

Train a certified 3-layer Transformers:
```
export PYTHONPATH=$PYTHONPATH:libs/xu_auto_LiRPA

python libs/xu_auto_LiRPA/examples/language/train.py \
  --dir=model_data/transformer_cert --robust \
  --method=IBP+backward_train --train --max_sent_length 128 \
  --num_layers 3
```

Attack 10 examples from the SST2 dataset.
```
./patched_textattack attack --attack-from-file=biasattack.py:SOBeamAttack \
  --dataset-from-nlp=glue:sst2:validation --num-examples=10 --shuffle=False \
  --model=models/xu_auto_LiRPA.py:transformer_cert
```

### Custom models

Our code is general and can be used to evaluate custom models. Similar to `models/jia_certified.py`, you will need to create a wrapper `models/custom_model.py` and implement two classes:
1. `class CustomTokenizer`
    - `def encode():`
    - Optional: `def batch_encode():`
2. `class ModelWrapper`
    - `def __call__():`
    - `def to():`

And then the model and tokenizer can be specified with `--model=models/custom_model.py:model_obj:tokenizer_obj`, where `model_obj` and `tokenizer_obj` are the variables of the corresponding type.

### Other models from TextAttack

Our code is built upon [Qdata/TextAttack](https://github.com/QData/TextAttack) and thus shares the similar API.

Attack a pre-trained model `lstm-sst2` in [TextAttack Model Zoo](https://github.com/chong-z/TextAttack/blob/d6ebeeb1afae215d7de5f04c3aac743bbeaf54db/textattack/models/README.md):
```
./patched_textattack attack --attack-from-file=biasattack.py:SOBeamAttack \
  --dataset-from-nlp=glue:sst2:validation --num-examples=10 --shuffle=False \
  --model=lstm-sst2
```

Train a `bert-base-uncased` with the `textattack train` command:
```
./patched_textattack train --model=bert-base-uncased --from-pretrained=True \
  --batch-size=32 --epochs=5 --learning-rate=2e-5 --seed=168 \
  --dataset=glue:sst2 --max-length=128 --save-last
```

The resulting model can be found under `model_data/sweeps/bert-base-uncased_pretrained_glue:sst2_no-aug_2021-04-08-14-00-49-733736`. To attack:
```
./patched_textattack attack --attack-from-file=biasattack.py:SOBeamAttack \
  --dataset-from-nlp=glue:sst2:validation --num-examples=10 --shuffle=False \
  --model=model_data/sweeps/bert-base-uncased_pretrained_glue:sst2_no-aug_2021-04-08-14-00-49-733736
```

## Attack Parameters

- `attack-from-file`: See `biasattack.py` for a list of algorithms.
    - `SOEnumAttack`: The brute-force SO-Enum attack that enumerates all neighborhood within distance `k=2`.
    - `SOBeamAttack`: The beam search based SO-Beam attack that searches within the neighborhood of distance `k=6`.
    - `RandomBaselineAttack`: The random baseline method mentioned in Appendix.
    - `BiasAnalysisChecklist`: The enumeration method used for evaluating the counterfactual bias on protected tokens from Ribeiro et al. (2020).
    - `BiasAnalysisGender`: The enumeration method used for evaluating the counterfactual bias on gendered pronounces from Zhao et al. (2018a).
- `dataset-from-nlp`: The name of the HuggingFace dataset. It's also possible to load a custom dataset with `--dataset-from-file=datasets.py:sst2_simple`.
- `model`: The target model for the attack. Can be a custom model in the form of `model_wrapper.py:model_obj:tokenizer_obj`, or the name/path of the TextAttack model.
- Additional parameters: Pleaser refer to `./patched_textattack --help` and `./patched_textattack attack --help`.

## Collect Metrics with `wandb`

We use `wandb` to collect metrics for both training and attacking. To enable `wandb`, please do the following:
1. Sign up for a free account through `wandb login`, or go to the [sign up page](https://app.wandb.ai/login?signup=true).
2. Append `--enable-wandb` to the training and attacking commands mentioned previously.

Please refer to https://docs.wandb.ai/quickstart for detailed guides.

## Credits
1. `TextAttack`: https://github.com/QData/TextAttack.
2. `libs/jia_certified`: https://github.com/robinjia/certified-word-sub.
3. `libs/xu_auto_LiRPA`: https://github.com/KaidiXu/auto_LiRPA.
4. See paper for the full list of references.
