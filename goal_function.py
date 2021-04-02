import torch
import lru
import numpy as np

from cached_property import cached_property

from textattack.goal_function_results import GoalFunctionResult
from textattack.goal_function_results.goal_function_result import GoalFunctionResultStatus
from textattack.goal_functions import GoalFunction
from textattack.shared import utils

from language_model import LanguageModelScorer
from utils import find_words_in_list, generate_biased_texts, replace_word


class BiasGoalFunctionResult(GoalFunctionResult):
    """
    Represents the result of a classification goal function.
    """

    # Initialized by BiasGoalFunction
    lm = None

    def _processed_output(self, color_method):
        assert 'active_biaswords' in self.attacked_text.attack_attrs, 'must have attack_attrs[`active_biaswords`], see utils.generate_biased_texts()'
        active_biaswords = self.attacked_text.attack_attrs['active_biaswords']
        colored_words = self._list_to_colored_string(active_biaswords,
                                                     self.raw_output.argmax(),
                                                     self.raw_output.argmin(),
                                                     color_method)
        ids = find_words_in_list(self.attacked_text.words, active_biaswords)
        return self.attacked_text.replace_words_at_indices(
            ids, [colored_words] *
            len(ids)).printable_text(key_color_method=color_method)

    def get_text_color_input(self):
        return "red"

    def get_text_color_perturbed(self):
        return "blue"

    def get_colored_output(self, color_method=None):
        output = self._processed_output(color_method=color_method)
        colored_scores = self._list_to_colored_string(self.raw_output.numpy(),
                                                      self.raw_output.argmax(),
                                                      self.raw_output.argmin(),
                                                      color_method)
        output_str = f"\n{output} (diff: {self.bias_diff:.6f}, preds: {colored_scores}, lm_scores: {self.lm_scores}"
        if self.num_queries <= 1:
            # Init result.
            output_str += f", truth:{self.ground_truth_output}"
        output_str += ")\n"

        return output_str

    def _list_to_colored_string(self, l, max_id, min_id, color_method):
        colored_l = [str(e) for e in l]
        colored_l[max_id] = utils.color_text(colored_l[max_id],
                                             color="green",
                                             method=color_method)
        colored_l[min_id] = utils.color_text(colored_l[min_id],
                                             color="purple",
                                             method=color_method)
        return '/'.join(colored_l)

    @cached_property
    def bias_diff(self):
        return self.raw_output.max().item() - self.raw_output.min().item()

    @cached_property
    def lm_score(self):
        return self.lm.get_text_score(self.attacked_text)

    @cached_property
    def lm_scores(self):
        biased_texts = generate_biased_texts(self.attacked_text)
        scores = [f'{self.lm.get_text_score(t):.6f}' for t in biased_texts]
        return '/'.join(scores)


class CompositeGoalFunction(GoalFunction):

    def __init__(
        self,
        goal_functions,
    ):
        self.goal_functions = goal_functions
        self.maximizable = False

    def init_attack_example(self, attacked_text, ground_truth_output):
        for g in self.goal_functions:
            g.init_attack_example(attacked_text, ground_truth_output)
        self.ground_truth_output = ground_truth_output
        result, _ = self.get_result(attacked_text, check_skip=True)
        return result, _

    def _is_goal_complete(self, *args):
        raise NotImplementedError()

    def _get_score(self, *args):
        raise NotImplementedError()

    def _goal_function_result_type(self):
        raise NotImplementedError()

    def _process_model_outputs(self, inputs, outputs):
        raise NotImplementedError()

    def get_output(self, attacked_text):
        return self.goal_functions[0].get_output(attacked_text)

    def get_results(self, attacked_text_list, check_skip=False):
        composite_results, composite_over = self.goal_functions[0].get_results(
            attacked_text_list, check_skip)

        for i in range(1, len(self.goal_functions)):
            g = self.goal_functions[i]
            results, over = g.get_results(attacked_text_list, check_skip)
            for j in range(len(results)):
                composite_results[j].score = min(composite_results[j].score,
                                                 results[j].score)
                if composite_results[
                        j].goal_status == GoalFunctionResultStatus.SUCCEEDED and results[
                            j].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    composite_results[
                        j].goal_status = GoalFunctionResultStatus.SUCCEEDED
                else:
                    composite_results[
                        j].goal_status = GoalFunctionResultStatus.SEARCHING
            composite_over = composite_over or over

        return composite_results, composite_over

    @property
    def query_budget(self):
        return np.sum([g.query_budget for g in self.goal_functions])

    @property
    def num_queries(self):
        return np.sum([g.num_queries for g in self.goal_functions])

    def extra_repr_keys(self):
        return ["goal_functions"]


class BiasGoalFunction(GoalFunction):

    # score_mode:
    #       - "ordered": Calculate bias score with `biaswords[0] - biaswords[1]`.
    #       - "reversed": Calculate bias score with `biaswords[1] - biaswords[0]`.
    #       - "max": Calculate bias score with `biaswords.max() - biaswords.min()`.
    def __init__(
        self,
        model,
        biaswords_list,
        active_biaswords_logit_threshold=2,
        biasthreshold=0.6,
        diffthreshold=0.05,
        stepweight=0.1,
        skipthreshold=0.1,
        score_mode="max",
        lm_scorer="gpt2",
        maximizable=False,
        tokenizer=None,
        use_cache=True,
        query_budget=float("inf"),
        model_batch_size=32,
        model_cache_size=2**20,
    ):
        assert biasthreshold >= 0.5, "`biasthreshold` must be >= 0.5"
        self.biasthreshold = biasthreshold
        self.diffthreshold = diffthreshold
        self.stepweight = stepweight
        self.skipthreshold = 0.1
        self.score_mode = score_mode
        self.lm_scorer = lm_scorer
        self.lm = LanguageModelScorer(language_model_name=lm_scorer)
        BiasGoalFunctionResult.lm = self.lm
        # # Ignore the validator.
        # validators.validate_model_goal_function_compatibility(
        #     self.__class__, model.__class__
        # )
        self.model = model
        self.model.eval()
        self.biaswords_list = biaswords_list
        self.biaswords_list_len = len(biaswords_list)
        for biaswords in biaswords_list:
            assert len(biaswords) == 2, "only supports pairs of biaswords"
        self.active_biaswords_logit_threshold = active_biaswords_logit_threshold
        self.biaswords_pair_len = 2
        self.maximizable = maximizable
        self.tokenizer = tokenizer
        if not self.tokenizer:
            if hasattr(self.model, "tokenizer"):
                self.tokenizer = self.model.tokenizer
            else:
                raise NameError(
                    "Cannot instantiate goal function without tokenizer")
        if not hasattr(self.tokenizer, "encode"):
            raise TypeError("Tokenizer must contain `encode()` method")
        self.use_cache = use_cache
        self.query_budget = query_budget
        self.model_batch_size = model_batch_size
        if self.use_cache:
            self._call_model_cache = lru.LRU(model_cache_size)
        else:
            self._call_model_cache = None

    def init_attack_example(self, attacked_text, ground_truth_output):
        attacked_text.attack_attrs[
            'active_biaswords'] = self._get_active_biaswords(attacked_text)
        return super().init_attack_example(attacked_text, ground_truth_output)

    def _call_model_uncached(self, attacked_text_list):
        biased_text_list = []
        for t in attacked_text_list:
            biased_text_list.extend(self._generate_biased_texts(t))

        return super()._call_model_uncached(biased_text_list)

    def _process_model_outputs(self, inputs, scores):
        classification_scores = self._compute_classification_score(
            inputs, scores)
        assert classification_scores.shape[
            1] == 2, "Currently only supports binary classification"

        label_one_scores = classification_scores[:, 1]
        return label_one_scores.view(-1, self.biaswords_pair_len)

    def _ordered_outputs(self, model_output):
        if self.score_mode == 'ordered':
            return (model_output[0], model_output[1])
        elif self.score_mode == 'reversed':
            return (model_output[1], model_output[0])
        elif self.score_mode == 'max':
            return (model_output.max(), model_output.min())
        raise ValueError(f"Unsupported score_mode:{self.score_mode}.")

    def _get_score(self, model_output, attacked_text):

        def step_func(x):
            if x <= self.biasthreshold:
                return x
            return self.biasthreshold + self.stepweight * (x -
                                                           self.biasthreshold)

        goal_complete_bonus = 100 if self._is_goal_complete(
            model_output, attacked_text) else 0

        ordered_outputs = self._ordered_outputs(model_output)
        return goal_complete_bonus + np.log(
            ordered_outputs[0]) + np.log(1 - ordered_outputs[1])

    def _compute_classification_score(self, inputs, scores):
        """ Forked from ClassificationGoalFunction.
        """
        # Validation check on model score dimensions
        if scores.ndim == 1:
            # Unsqueeze prediction, if it's been squeezed by the model.
            if len(inputs == 1):
                scores = scores.unsqueeze(dim=0)
            else:
                raise ValueError(
                    f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
                )
        elif scores.ndim != 2:
            # If model somehow returns too may dimensions, throw an error.
            raise ValueError(
                f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
            )
        elif not ((scores.sum(dim=1) - 1).abs() < 1e-6).all():
            # Values in each row should sum up to 1. The model should return a
            # set of numbers corresponding to probabilities, which should add
            # up to 1. Since they are `torch.float` values, allow a small
            # error in the summation.
            scores = torch.nn.functional.softmax(scores, dim=1)
            if not ((scores.sum(dim=1) - 1).abs() < 1e-6).all():
                raise ValueError("Model scores do not add up to 1.")
        return scores.cpu()

    def _is_goal_complete(self, model_output, attacked_text):
        ordered_outputs = self._ordered_outputs(model_output)
        return ordered_outputs[0] >= self.biasthreshold and ordered_outputs[
            1] <= (1 - self.biasthreshold) and (
                ordered_outputs[0] - ordered_outputs[1]) >= self.diffthreshold

    def _should_skip(self, model_output, attacked_text):
        active_biaswords = attacked_text.attack_attrs['active_biaswords']
        if active_biaswords[0] not in attacked_text.words and active_biaswords[
                1] not in attacked_text.words:
            return True
        return False

    def _goal_function_result_type(self):
        return BiasGoalFunctionResult

    def _get_displayed_output(self, raw_output):
        return raw_output

    def _generate_biased_texts(self, input_text):
        return generate_biased_texts(input_text)

    def _get_active_biaswords(self, input_text):
        lm_score_threshold = self.lm.get_text_score(
            input_text) - self.active_biaswords_logit_threshold

        is_long_text = len(input_text.words) > 20

        def is_good_biaswords(biaswords):
            l = list(biaswords)
            assert len(l) == 2
            for w in l:
                if input_text.words.count(w) == 1:
                    l.remove(w)
                    if l[0] in input_text.words:
                        return False
                    if is_long_text:
                        # Skip the lm_score check due to the long running time.
                        return True
                    ref_text = replace_word(input_text, w, l[0])
                    ref_lm_score = self.lm.get_text_score(ref_text)
                    if ref_lm_score >= lm_score_threshold:
                        return True
            return False

        for biaswords in self.biaswords_list:
            if is_good_biaswords(biaswords):
                return biaswords
        # TODO: Better error handling.
        return ['[NULL]', '[NULL]']

    def extra_repr_keys(self):
        return [
            "biaswords_list_len", "active_biaswords_logit_threshold",
            "biasthreshold", "diffthreshold", "stepweight", "skipthreshold",
            "score_mode", "lm_scorer", "model_batch_size",
            *super().extra_repr_keys()
        ]
