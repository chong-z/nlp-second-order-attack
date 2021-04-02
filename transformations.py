import numpy as np
import torch
import platform
from mezmorize import Cache

import textattack
from textattack.transformations import Transformation, WordSwapMaskedLM

import utils


class Identity(Transformation):

    def _get_transformations(self, current_text, indices_to_modify):
        return [current_text]


class InitialBiasWord(Transformation):

    def __init__(self, biaswords_flatten):
        self.biaswords_flatten = biaswords_flatten
        self.biaswords_flatten_len = len(biaswords_flatten)

    def _get_replacement_words(self, word):
        return self.biaswords_flatten

    def _get_transformations(self, current_text, indices_to_modify):
        # Don't replace if |current_text| already contains |self.biaswords_flatten|.
        ids = utils.find_words_in_list(current_text.words,
                                       self.biaswords_flatten)
        if len(ids) > 0:
            return []

        words = current_text.words
        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = words[i]
            replacement_words = self._get_replacement_words(word_to_replace)
            transformed_texts_idx = []
            for r in replacement_words:
                if r == word_to_replace:
                    continue
                transformed_texts_idx.append(
                    current_text.replace_word_at_index(i, r))
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts

    def extra_repr_keys(self):
        return ["biaswords_flatten_len", *super().extra_repr_keys()]


class WordSwapMaskedLMPlus(WordSwapMaskedLM):

    def __init__(self,
                 *args,
                 max_masks,
                 max_trials,
                 logit_threshold,
                 force_rte_format=False,
                 **kwargs):
        self.max_masks = max_masks
        self.max_trials = max_trials
        self.logit_threshold = logit_threshold
        self.force_rte_format = force_rte_format

        diskcache_config = {
            'CACHE_TYPE': 'filesystem',
            'CACHE_DEFAULT_TIMEOUT': 7 * 24 * 3600,
            'CACHE_THRESHOLD': 2 * 2**20,
            'CACHE_DIR': f'mezmorize_cache-py{platform.python_version()}',
        }

        self.diskcache = Cache(**diskcache_config)
        super().__init__(*args, method="bae_plus", **kwargs)

    def _get_replacement_words_diskcached(self, current_text, trials):
        """Get replacement words for the word we want to replace using BAE
        method.

        Args:
            current_text (AttackedText): Text we want to get replacements for.
            trials ([[int]]): list of indices of word we want to replace
        """

        def make_key(text):
            return f'{self}-{text}'

        masked_attacked_texts = [
            current_text.replace_words_at_indices(
                indices, [self._lm_tokenizer.mask_token] * len(indices)).text
            for indices in trials
        ]

        all_outputs = [None] * len(trials)
        uncached_texts = []
        uncached_indices = []
        for i, text in enumerate(masked_attacked_texts):
            output = self.diskcache.get(make_key(text))
            if output is None:
                uncached_texts.append(text)
                uncached_indices.append(i)
            else:
                all_outputs[i] = output

        new_outputs = self._get_replacement_words_uncached(uncached_texts)
        for i, text, output in zip(uncached_indices, uncached_texts,
                                   new_outputs):
            self.diskcache.set(make_key(text), output)
            all_outputs[i] = output

        assert None not in all_outputs

        return all_outputs

    def _get_replacement_words_uncached(self, masked_texts):
        if len(masked_texts) == 0:
            return []

        batch_inputs = self._batch_encode(masked_texts)

        with torch.no_grad():
            batch_preds = self._language_model(**batch_inputs)[0]

        batch_ids = batch_inputs["input_ids"]
        return [
            self._filtered_top_words(preds, ids)
            for preds, ids in zip(batch_preds, batch_ids)
        ]

    def _filtered_top_words(self, preds, ids):
        masked_indices = torch.where(ids == self._lm_tokenizer.mask_token_id)[0]
        if len(masked_indices) == 0:
            return []

        mask_token_probs = preds[masked_indices]
        topk = torch.topk(mask_token_probs, self.max_candidates)

        top_logits = topk[0].T
        top_ids = topk[1].T

        filtered_ids = top_ids[(top_logits > top_logits[0] -
                                self.logit_threshold).all(dim=1)].tolist()

        replacement_words = []
        for new_ids in filtered_ids:
            new_tokens = self._lm_tokenizer.convert_ids_to_tokens(new_ids)
            if are_one_word(new_tokens) and check_no_subwords(new_tokens):
                replacement_words.append(new_tokens)

        return replacement_words

    def _get_transformations(self, current_text, indices_to_modify):
        if len(indices_to_modify) == 0:
            return []

        transformed_texts = []

        num_masks = min(len(current_text.words), self.max_masks)

        if self.max_trials == -1:
            assert num_masks == 1, "num_masks must be 1 when max_trials == -1"
            trials = [[i] for i in indices_to_modify]
        else:
            allowed_indices = list(indices_to_modify)
            # Each trial may contain [1, num_masks] masks, in sorted order.
            trials = np.array([
                np.random.choice(allowed_indices, num_masks, replace=True)
                for i in range(self.max_trials)
            ])
            trials.sort(axis=1)
            trials = np.unique(trials, axis=0)
            trials = [np.unique(t) for t in trials]

        replacement_words_per_trial = self._get_replacement_words_diskcached(
            current_text, trials)

        for indices, replacement_words in zip(trials,
                                              replacement_words_per_trial):
            words_at_indices = np.array(current_text.words)[indices]
            transformed_texts_idx = []
            for r in replacement_words:
                if np.any(r != words_at_indices):
                    transformed_texts_idx.append(
                        current_text.replace_words_at_indices(indices, r))
            transformed_texts.extend(transformed_texts_idx)

        if self.force_rte_format and len(current_text.column_labels) == 2:
            tmp = []
            for text in transformed_texts:
                tmp.append(force_rte_format(text))
            transformed_texts = tmp

        return transformed_texts

    def _batch_encode(self, texts):
        encoding = self._lm_tokenizer.batch_encode_plus(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            k: v.to(textattack.shared.utils.device)
            for k, v in encoding.items()
        }

    def extra_repr_keys(self):
        return [
            "max_masks", "max_trials", "logit_threshold", "force_rte_format",
            *super().extra_repr_keys()
        ]


def are_one_word(tokens):
    return np.all([textattack.shared.utils.is_one_word(w) for w in tokens])


def check_no_subwords(tokens):
    return not np.any([check_if_subword(w) for w in tokens])


def check_if_subword(token):
    return True if "##" in token else False


# Make sure the text is in the form:
# `The A ... the B ... she/he [....]\nThe A/B [....]`.
def force_rte_format(input_text):
    assert len(input_text.column_labels) == 2
    active_biaswords = utils.get_active_biaswords(input_text)
    assert len(active_biaswords) == 2
    premise, hypothesis = input_text.tokenizer_input
    premise_words, hypothesis_words = input_text.words_per_input
    biasword = active_biaswords[0] if active_biaswords[
        0] in premise_words else active_biaswords[1]
    assert biasword in premise_words
    after_biasword = premise[premise.find(biasword) + len(biasword):]
    new_hypothesis = ' '.join(hypothesis_words[:2]) + after_biasword
    new_words = textattack.shared.utils.words_from_text("\n".join(
        [premise, new_hypothesis]))
    return input_text.generate_new_attacked_text(new_words)
