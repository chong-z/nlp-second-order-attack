import torch
from torch import nn as nn
from torch.nn import functional as F
from unittest.mock import patch

import utils
import textattack

xu_utils = utils.import_module_from_path(
    module_name='libs.xu_auto_LiRPA.examples.language.Transformer.utils',
    module_path='libs/xu_auto_LiRPA/examples/language')


def get_xu_model(load_path='model_data/transformer_cert/ckpt_3',
                 device=textattack.shared.utils.device):
    args = f'''train.py
    --load={load_path}
    --max_sent_length=128
    --num_layers=3
    --robust
    --method=IBP+backward
    --dir=.cache
    --device={device.type}'''.split()

    with patch('argparse._sys.argv', args):
        train = utils.import_module_from_path(
            module_name='libs.xu_auto_LiRPA.examples.language.train')
        model = train.model
    return model


class XuTokenizer:

    def __init__(self):
        self.model = None

    def set_model(self, model):
        assert self.model is None
        self.model = model

    def _process_text(self, text_input):
        """A text input may be a single-input tuple (text,) or multi-input
        tuple (text, text, ...).

        In the single-input case, unroll the tuple. In the multi-input
        case, raise an error.
        """
        if isinstance(text_input, tuple):
            if len(text_input) > 1:
                raise ValueError(
                    "Cannot use `GloveTokenizer` to encode multiple inputs")
            text_input = text_input[0]
        return text_input

    def encode(self, text):
        return self.batch_encode([text])[0]

    def batch_encode(self, input_text_list):
        return self.batch_encode_with_size(self, input_text_list, 1)

    def batch_encode_with_size(self, input_text_list, batch_size):
        batch = [{
            'sentence': self._process_text(text),
            'label': 0,
        } for text in input_text_list]

        features = xu_utils.convert_examples_to_features(
            batch,
            self.model.model.label_list,
            self.model.model.max_seq_length,
            self.model.model.vocab,
            drop_unk=self.model.model.drop_unk)

        outputs = []
        i = 0
        while i < len(features):
            batch_features = features[i:i + batch_size]

            input_ids = torch.tensor([f.input_ids for f in batch_features],
                                     dtype=torch.long).to(
                                         self.model.model.device)
            input_mask = torch.tensor([f.input_mask for f in batch_features],
                                      dtype=torch.long).to(
                                          self.model.model.device)
            segment_ids = torch.tensor([f.segment_ids for f in batch_features],
                                       dtype=torch.long).to(
                                           self.model.model.device)

            outputs.append({
                'input_ids': input_ids,
                'token_type_ids': segment_ids,
                'attention_mask': input_mask,
            })
            i += batch_size

        return outputs


class ModelWrapper:

    def __init__(self, load_path, tokenizer):
        self.model = None
        self.load_path = load_path
        self.tokenizer = tokenizer

    def __call__(self, **kwargs):
        with torch.no_grad():
            logits = self.model.model(**kwargs)
        return logits

    def to(self, device):
        # Guaranteed to be called by textattack before using. We do
        # setup works here.
        assert self.model is None
        self.tokenizer.set_model(self)

        self.model = get_xu_model(load_path=self.load_path, device=device)
        self.model.eval()
        return self

    def eval(self):
        self.model.eval()


# Supported models
tokenizer = XuTokenizer()

# INFO     2020-11-13 21:59:51,092 Epoch 3, train step 2105/2105: eps 0.00000, acc=0.9000 loss=0.2430 acc_rob=0.9000 loss_rob=0.2430
# INFO     2020-11-13 21:59:51,798 Epoch 3, dev step 28/28: eps 0.00000, acc=0.8211 loss=0.4239 acc_rob=0.8211 loss_rob=0.4239
# INFO     2020-11-13 21:59:53,208 Epoch 3, test step 57/57: eps 0.00000, acc=0.8045 loss=0.4271 acc_rob=0.8045 loss_rob=0.4271
transformer_normal = ModelWrapper(
    load_path='model_data/transformer_normal/ckpt_3', tokenizer=tokenizer)

# INFO     2020-11-13 20:07:59,383 Epoch 3, train step 2105/2105: eps 0.29995, acc=0.8801 loss=0.2836 acc_rob=0.8475 loss_rob=0.3417
# INFO     2020-11-13 20:08:02,968 Epoch 3, dev step 28/28: eps 0.29995, acc=0.8188 loss=0.4076 acc_rob=0.7718 loss_rob=0.4702
# INFO     2020-11-13 20:08:10,565 Epoch 3, test step 57/57: eps 0.29995, acc=0.8100 loss=0.4140 acc_rob=0.7721 loss_rob=0.4731
transformer_cert = ModelWrapper(
    load_path='model_data/transformer_cert/ckpt_3', tokenizer=tokenizer)

