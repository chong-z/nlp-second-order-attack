import torch
import numpy as np

from lm_scorer.models.auto import AutoLMScorer as LMScorer
from transformers import AutoModelForMaskedLM, AutoTokenizer

from textattack.shared import utils


class LanguageModelScorer:

    def __init__(
        self,
        language_model_name="gpt2",
        batch_size=1,
    ):
        self._lm_scorer = None
        self._masked_lm_scorer = None
        if language_model_name in LMScorer.supported_model_names():
            self._lm_scorer = LMScorer.from_pretrained(language_model_name,
                                                       device=utils.device,
                                                       batch_size=batch_size)
        else:
            self._masked_lm_scorer = MaskedLanguageModelScorer(
                masked_language_model=language_model_name,
                batch_size=batch_size)

    def get_text_score(self, attacked_text):
        if self._lm_scorer is not None:
            return self._lm_scorer.sentence_score(attacked_text.text,
                                                  reduce='gmean',
                                                  log=True)
        return self._masked_lm_scorer.get_text_score(attacked_text)


class MaskedLanguageModelScorer:

    def __init__(
        self,
        masked_language_model="bert-base-uncased",
        batch_size=1,
        max_length=256,
    ):
        self.masked_lm_name = masked_language_model
        self.max_length = max_length

        self._lm_tokenizer = AutoTokenizer.from_pretrained(
            masked_language_model, use_fast=True)
        self._language_model = AutoModelForMaskedLM.from_pretrained(
            masked_language_model)
        self._language_model.to(utils.device)
        self._language_model.eval()

    def get_text_score(self, attacked_text):
        return np.mean(
            np.log([
                self._get_prob_at_index(attacked_text, i)
                for i in range(len(attacked_text.words))
            ]))

    def _get_prob_at_index(self, text, index):
        masked_attacked_text = text.replace_word_at_index(
            index, self._lm_tokenizer.mask_token)
        inputs = self._encode_text(masked_attacked_text.text)
        masked_token_id = self._lm_tokenizer.convert_tokens_to_ids(
            text.words[index])
        ids = inputs["input_ids"].tolist()[0]

        try:
            # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
            masked_index = ids.index(self._lm_tokenizer.mask_token_id)
        except ValueError:
            return -1

        with torch.no_grad():
            preds = self._language_model(**inputs)[0]

        masked_token_logit = preds[0, masked_index]
        masked_token_probs = torch.nn.functional.softmax(masked_token_logit,
                                                         dim=0)
        return masked_token_probs[masked_token_id].item()

    def _encode_text(self, text):
        """Encodes ``text`` using an ``AutoTokenizer``, ``self._lm_tokenizer``.

        Returns a ``dict`` where keys are strings (like 'input_ids') and
        values are ``torch.Tensor``s. Moves tensors to the same device
        as the language model.
        """
        encoding = self._lm_tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.to(utils.device) for k, v in encoding.items()}


def gmean(input_x):
    log_x = np.log(input_x)
    return np.exp(np.mean(log_x))
