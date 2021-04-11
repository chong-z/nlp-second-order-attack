import re

import numpy as np
import pandas as pd

import utils

from textattack.datasets import TextAttackDataset, HuggingFaceNlpDataset


class SST2Dataset(TextAttackDataset):

    def __init__(self, text_file_name, duplicate_map=None, replace_xy=False):
        self.text_file_name = text_file_name
        self.duplicate_map = duplicate_map
        self.replace_xy = replace_xy
        self.neutralize_pairs = None
        if replace_xy:
            self.x_words = set()
            self.y_words = set()
            gender_swap_words, _ = utils.get_gender_swap_words()
            # |gender_swap_words| may contain reversed pairs and we use
            # the first appearance.
            for w1, w2 in gender_swap_words:
                if w1 not in self.x_words and w1 not in self.y_words:
                    self.x_words.add(w1)
                if w2 not in self.x_words and w2 not in self.y_words:
                    self.y_words.add(w2)

    def setup(self, num_examples_offset, shuffle, attack_args):
        self._maybe_parse_biaswords_list(attack_args)

        self._load_classification_text_file(self.text_file_name,
                                            offset=num_examples_offset,
                                            shuffle=shuffle)

    def _maybe_parse_biaswords_list(self, attack_args):
        if attack_args is None or ":" not in attack_args:
            return

        args = attack_args.split(":")
        if len(args) == 3:
            biaswords_list = utils.biaswords_list_from_str(args[2])
            self.duplicate_map = biaswords_list
            self._maybe_setup_gender_neutralize(biaswords_list)

    def _maybe_setup_gender_neutralize(self, biaswords_list):
        should_neutralize_gender = False
        biaswords_flatten = utils.flatten_nested_list(biaswords_list)

        gender_define_words, _ = utils.get_gender_define_words()
        for w in biaswords_flatten:
            if w in gender_define_words:
                should_neutralize_gender = True
                break

        if should_neutralize_gender:
            self.neutralize_pairs = utils.get_gender_neutralize_pairs()
            self.neutralize_pairs = [
                p for p in self.neutralize_pairs
                if p[0] not in biaswords_flatten
            ]

    def _replace_words(self, text, label, replacement_pairs):
        for w1, w2 in replacement_pairs:
            text = re.sub(r"([^a-zA-Z]|^)" + w1 + r"([^a-zA-Z]|$)",
                          f"\\1{w2}\\2", text)
        return (text, label)

    def _duplicate_example(self, text, label):
        words = text.split()
        for w1, w2 in self.duplicate_map:
            if w1 in words or w2 in words:
                return [
                    (re.sub(r"([^a-zA-Z]|^)" + w2 + f"([^a-zA-Z]|$)",
                            f"\\1{w1}\\2", text), label),
                    (re.sub(r"([^a-zA-Z]|^)" + w1 + f"([^a-zA-Z]|$)",
                            f"\\1{w2}\\2", text), label),
                ]
        return []

    def _replace_xy(self, text, label):
        assert self.replace_xy

        for w in self.x_words:
            text = re.sub(r"([^a-zA-Z]|^)" + w + f"([^a-zA-Z]|$)", f"\\1x\\2",
                          text)

        for w in self.y_words:
            text = re.sub(r"([^a-zA-Z]|^)" + w + f"([^a-zA-Z]|$)", f"\\1y\\2",
                          text)

        return (text.strip(), label)

    def _load_classification_text_file(self,
                                       text_file_name,
                                       offset=0,
                                       shuffle=False):
        pd_data = pd.read_csv(text_file_name, sep='\t')
        self.examples = pd_data.to_numpy()

        if self.duplicate_map is not None:
            duplicated_examples = []
            for text, label in self.examples:
                duplicated_examples.extend(self._duplicate_example(text, label))
            self.examples = duplicated_examples

        if self.neutralize_pairs is not None:
            self.examples = [
                self._replace_words(text, label, self.neutralize_pairs)
                for text, label in self.examples
            ]

        if self.replace_xy:
            self.examples = [
                self._replace_xy(text, label) for text, label in self.examples
            ]
        self._i = 0
        self.examples = self.examples[offset:]
        if shuffle:
            np.random.shuffle(self.examples)


class SimpleDataset(TextAttackDataset):

    def __init__(self):
        self.examples = [
            ('we root for clara and paul , even like them , though perhaps it ’s an emotion closer to pity .',
             1),
        ]

    def setup(self, num_examples_offset, shuffle, attack_args):
        pass


class HuggingFaceNlpDatasetPlus(HuggingFaceNlpDataset):

    def __init__(self, replace_map=[], **kwargs):
        self.replace_map = replace_map
        # Defer setup.
        self.kwargs = kwargs

    def setup(self, num_examples_offset, shuffle, attack_args):
        super().__init__(**self.kwargs)

        self.examples = self.examples[num_examples_offset:]
        if shuffle:
            np.random.shuffle(self.examples)

    def _format_raw_example(self, raw_example):
        for w1, w2 in self.replace_map:
            raw_example['premise'] = re.sub(
                r"([^a-zA-Z]|^)" + w1 + f"([^a-zA-Z]|$)", f"\\1{w2}\\2",
                raw_example['premise'])
        return super()._format_raw_example(raw_example)


sst2_dev = SST2Dataset("SST-2/dev.tsv")
sst2_train = SST2Dataset("SST-2/train.tsv")

sst2_simple = SimpleDataset()
