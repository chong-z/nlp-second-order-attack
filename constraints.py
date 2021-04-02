from textattack.constraints import Constraint, PreTransformationConstraint
from textattack.shared.validators import transformation_consists_of_word_swaps

from transformations import Identity, InitialBiasWord

from language_model import LanguageModelScorer


class WordBlacklistConstraint(Constraint):

    def __init__(self, *args, blacklist_words, **kwargs):
        self.blacklist_words = blacklist_words
        self.num_blacklist = len(blacklist_words)
        super().__init__(*args, compare_against_original=False, **kwargs)

    def _check_constraint(self, transformed_text, reference_text):
        # Allow word deletion.
        if len(transformed_text.words) < len(reference_text.words):
            return True

        # Allow Identity and InitialBiasWord transformation.
        last_transformation = transformed_text.attack_attrs[
            "last_transformation"]
        if isinstance(last_transformation, Identity) or isinstance(
                last_transformation, InitialBiasWord):
            return True

        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply word blacklist constraint without `newly_modified_indices`"
            )

        for i in indices:
            if transformed_text.words[i] in self.blacklist_words:
                return False
        return True

    def extra_repr_keys(self):
        return ["num_blacklist", *super().extra_repr_keys()]


class ActiveAnchorWordsModification(Constraint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, compare_against_original=False, **kwargs)

    def _check_constraint(self, transformed_text, reference_text):
        # Allow word deletion.
        if len(transformed_text.words) < len(reference_text.words):
            return True

        # Allow Identity and InitialBiasWord transformation.
        last_transformation = transformed_text.attack_attrs[
            "last_transformation"]
        if isinstance(last_transformation, Identity) or isinstance(
                last_transformation, InitialBiasWord):
            return True

        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
            active_biaswords = reference_text.attack_attrs['active_biaswords']
        except KeyError:
            raise KeyError(
                "Cannot apply active anchor words constraint without `newly_modified_indices` or `active_biaswords`"
            )

        for i in indices:
            if transformed_text.words[i] in active_biaswords or reference_text.words[i] in active_biaswords:
                return False
        return True


class LanguageModelScoreConstraint(Constraint):

    def __init__(self,
                 lm_scorer="gpt2",
                 score_ratio_threshold=0.5,
                 compare_against_original=True):
        super().__init__(compare_against_original)
        self.lm_scorer = lm_scorer
        self.lm = LanguageModelScorer(language_model_name=lm_scorer)
        self.score_ratio_threshold = score_ratio_threshold

    def _check_constraint(self, transformed_text, reference_text):
        original_score = self.lm.get_text_score(reference_text)
        score_ratio = self.lm.get_text_score(transformed_text) / original_score
        return score_ratio >= self.score_ratio_threshold

    def extra_repr_keys(self):
        return ["score_ratio_threshold", "lm_scorer"
               ] + super().extra_repr_keys()
