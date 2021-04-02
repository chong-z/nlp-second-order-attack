import torch
from torch import nn as nn
from torch.nn import functional as F
from unittest.mock import patch

import utils
import textattack

train = utils.import_module_from_path(
    module_name='libs.jia_certified.src.train')
text_classification = utils.import_module_from_path(
    module_name='libs.jia_certified.src.text_classification')
data_util = utils.import_module_from_path(
    module_name='libs.jia_certified.src.data_util')
device = textattack.shared.utils.device


class JiaTokenizer:

    def __init__(self,
                 vocab,
                 device,
                 dataset_cls=text_classification.TextClassificationDataset):
        self.vocab = vocab
        self.dataset_cls = dataset_cls
        self.device = device

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
        text = self._process_text(text)
        dataset = self.dataset_cls.from_raw_data([(text, 0)],
                                                 self.vocab,
                                                 attack_surface=None)
        data = dataset.get_loader(1)
        return data_util.dict_batch_to_device(next(iter(data)), self.device)

    def batch_encode(self, input_text_list):
        return self.batch_encode_with_size(self, input_text_list, 1)

    def batch_encode_with_size(self, input_text_list, batch_size):
        """The batch equivalent of ``encode``."""
        input_text_list = list(map(self._process_text, input_text_list))
        input_text_list = zip(input_text_list, [0] * len(input_text_list))

        dataset = self.dataset_cls.from_raw_data(input_text_list,
                                                 self.vocab,
                                                 attack_surface=None)
        inputs = dataset.examples

        # Do batch here since ibp.DiscreteChoiceTensor does not support slicing.
        outputs = []
        i = 0
        while i < len(inputs):
            batch = inputs[i:i + batch_size]
            batch_data = self.dataset_cls.collate_examples(batch)
            outputs.append(
                data_util.dict_batch_to_device(batch_data, self.device))
            i += batch_size

        return outputs


class ModelWrapper:

    def __init__(self, OPTS, vocab, word_mat):
        self.model = None
        self.OPTS = OPTS
        self.vocab = vocab
        self.word_mat = word_mat

    def __call__(self, **kwargs):
        with torch.no_grad():
            logits = self.model.forward(kwargs, compute_bounds=False)
            # Jia's model returns a single number as the output, where positive -> label 1
            # and negative -> label 0. We need to pad a single 0 before each row.
            outputs = F.pad(logits, (1, 0), 'constant', 0)
        return outputs

    def to(self, device):
        # Guaranteed to be called by textattack before using. We do
        # setup works here.
        if self.model is None:
            self.model = text_classification.load_model(self.word_mat, device,
                                                        self.OPTS)
        else:
            self.model = self.model.to(device)
        self.model.eval()
        return self

    def eval(self):
        self.model.eval()


def get_jia_args(load_path='model_data/cnn_cert', model_type='cnn'):
    args = f'''src/train.py classification {model_type}
    --out-dir outputs/eval_{load_path.replace('/', '_')}
    --data-cache-dir .cache
    --load-dir {load_path}
    --hidden-size 100 --pool mean --num-epochs 0
    '''.split()

    with patch('argparse._sys.argv', args):
        OPTS = train.parse_args()
    return OPTS


train_data, _, word_mat, _ = text_classification.load_datasets(
    device, get_jia_args('model_data/cnn_normal'))
vocab = train_data.vocab

# Default 'tokenizer' for '--model_from_file'.
tokenizer = JiaTokenizer(vocab=vocab, device=device)

# Supported models
cnn_normal = ModelWrapper(OPTS=get_jia_args('model_data/cnn_normal'),
                          vocab=vocab,
                          word_mat=word_mat)
cnn_cert = ModelWrapper(OPTS=get_jia_args('model_data/cnn_cert'),
                        vocab=vocab,
                        word_mat=word_mat)
cnn_cert_test = ModelWrapper(OPTS=get_jia_args('model_data/cnn_cert_test'),
                             vocab=vocab,
                             word_mat=word_mat)

bow_normal = ModelWrapper(OPTS=get_jia_args('model_data/bow_normal',
                                            model_type='bow'),
                          vocab=vocab,
                          word_mat=word_mat)
bow_cert = ModelWrapper(OPTS=get_jia_args('model_data/bow_cert',
                                          model_type='bow'),
                        vocab=vocab,
                        word_mat=word_mat)

lstm_normal = ModelWrapper(OPTS=get_jia_args('model_data/lstm_normal',
                                             model_type='lstm'),
                           vocab=vocab,
                           word_mat=word_mat)
lstm_cert = ModelWrapper(OPTS=get_jia_args('model_data/lstm_cert',
                                           model_type='lstm'),
                         vocab=vocab,
                         word_mat=word_mat)
