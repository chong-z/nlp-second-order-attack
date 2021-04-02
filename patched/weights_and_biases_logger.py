import os
import urllib
import torch
import json

from textattack.attack_results import FailedAttackResult, SkippedAttackResult, SuccessfulAttackResult, MaximizedAttackResult
from textattack.goal_function_results import GoalFunctionResult
from textattack.shared.utils import html_table_from_rows
from textattack.transformations import CompositeTransformation

from textattack.loggers import WeightsAndBiasesLogger as OriginalWeightsAndBiasesLogger

from transformations import WordSwapMaskedLMPlus
from goal_function import BiasGoalFunction, BiasGoalFunctionResult
from utils import extra_repr_dict, generate_biased_texts


class WeightsAndBiasesLogger(OriginalWeightsAndBiasesLogger):
    """Logs attack results to Weights & Biases."""

    def __init__(self, args, filename="", stdout=False):
        self.args = args
        self.extra_loggings = []
        super().__init__(filename=filename, stdout=stdout)

    def init_wandb(self):
        args = self.args

        tags = []
        env_tag = os.getenv('WANDB_TAG')
        if env_tag is not None and len(env_tag) > 0:
            tags.append(env_tag)
            del os.environ['WANDB_TAG']
        if self.args.wandb_tag is not None:
            tags.append(self.args.wandb_tag)

        model_name = args.model_from_file or args.model_from_huggingface or args.model
        dataset = args.dataset_from_file or args.dataset_from_nlp
        attack_method = args.recipe or args.attack_from_file

        import wandb
        self.wandb = wandb
        self.wandb.init(project="textattack", tags=tags, resume=False)
        self.wandb.config.update({
            "attack_method": args.attack_from_file,
            "model_name": model_name,
            "num_examples": args.num_examples,
            "num_examples_offset": args.num_examples_offset,
            "shuffle": args.shuffle,
            "dataset": dataset,
        })

    def log_summary_rows(self, rows, title, window_id):
        num_columns = len(rows[0])
        table = self.wandb.Table(columns=[title] +
                                 [f"C{i}" for i in range(1, num_columns)])
        for row in rows:
            table.add_data(*row)
        self.wandb.log({window_id: table})
        if num_columns == 2:
            for row in rows:
                metric_name, metric_score = row
                self.wandb.run.summary[metric_name] = metric_score

    def get_success_bias_result_log(self, result):
        assert type(result) in [SuccessfulAttackResult, MaximizedAttackResult]
        assert isinstance(result.original_result, BiasGoalFunctionResult)
        assert isinstance(result.perturbed_result, BiasGoalFunctionResult)

        removed_idxs, new_idxs = result.diff_idxs()

        removed_words = [
            result.original_result.attacked_text.words[i] for i in removed_idxs
        ]
        new_words = [
            result.perturbed_result.attacked_text.words[i] for i in new_idxs
        ]

        original_biased_inputs = [
            t.printable_text()
            for t in generate_biased_texts(result.original_result.attacked_text)
        ]
        perturbed_biased_inputs = [
            t.printable_text() for t in generate_biased_texts(
                result.perturbed_result.attacked_text)
        ]

        active_biaswords = result.perturbed_result.attacked_text.attack_attrs[
            'active_biaswords']

        return {
            "original_biased_inputs": original_biased_inputs,
            "perturbed_biased_inputs": perturbed_biased_inputs,
            "active_biaswords": active_biaswords,
            "removed_words": removed_words,
            "new_words": new_words,
        }

    def get_common_log(self, result):
        if type(result) not in [
                SuccessfulAttackResult, MaximizedAttackResult,
                FailedAttackResult
        ]:
            return {
                "attack_result_type": attack_result_type_to_str(type(result)),
            }

        assert isinstance(result.original_result, GoalFunctionResult)
        assert isinstance(result.perturbed_result, GoalFunctionResult)

        def format_output(result):
            output = result.output
            if isinstance(output, torch.Tensor):
                output = output.tolist()
            # Return |output| for second order attack, and |raw_output| for first order attacks.
            if type(output) == list:
                return output
            return result.raw_output

        return {
            "attack_result_type":
                attack_result_type_to_str(type(result)),
            "original_input":
                result.original_result.attacked_text.printable_text(),
            "original_outputs":
                format_output(result.original_result),
            "perturbed_input":
                result.perturbed_result.attacked_text.printable_text(),
            "perturbed_outputs":
                format_output(result.perturbed_result),
            "colored_result_ansi":
                urllib.parse.quote(result.__str__(color_method='ansi')),
            "colored_result_html":
                result.__str__(color_method='html'),
            "num_queries":
                result.perturbed_result.num_queries,
            "score":
                result.perturbed_result.score,
        }

    def log_extra_loggings(self):
        # extra_loggings exists the metric size limit.
        # wandb: ERROR Metric data exceeds maximum size of 4091904 bytes. Dropping it.
        # wandb: ERROR Summary data exceeds maximum size of 4091904 bytes. Dropping it.
        self.wandb.log({
            "extra_loggings":
                self.wandb.Html(json.dumps(self.extra_loggings), inject=False),
        })

    def log_attack_result(self, result):
        if not isinstance(result.original_result, BiasGoalFunctionResult):
            # Regular attack recipes.
            self.wandb.log(self.get_common_log(result))
            return

        # Log SuccessfulAttackResult
        if type(result) in [SuccessfulAttackResult, MaximizedAttackResult]:
            self.extra_loggings.append(result.perturbed_result.extra_logging)
            self.wandb.log({
                **self.get_common_log(result),
                "bias_diff":
                    result.perturbed_result.bias_diff,
                "lm_score_original":
                    result.original_result.lm_score,
                "lm_score_perturbed":
                    result.perturbed_result.lm_score,
                **self.get_success_bias_result_log(result),
            })
        elif type(result) == FailedAttackResult:
            self.extra_loggings.append(result.perturbed_result.extra_logging)
            self.wandb.log({
                **self.get_common_log(result),
                "bias_diff_failed":
                    result.perturbed_result.bias_diff,
                "active_biaswords":
                    result.perturbed_result.attacked_text.
                    attack_attrs['active_biaswords'],
            })
        else:
            self.wandb.log(self.get_common_log(result))

    def log_attack_details(self, attack, model):
        self.log_extra_loggings()

        masked_lm_transformation = None
        if isinstance(attack.transformation, CompositeTransformation):
            for transformation in attack.transformation.transformations:
                if isinstance(transformation, WordSwapMaskedLMPlus):
                    masked_lm_transformation = transformation
                    break
        if masked_lm_transformation is not None:
            self.wandb.config.update(
                extra_repr_dict(obj=masked_lm_transformation, prefix="trans_"))

        if isinstance(attack.goal_function, BiasGoalFunction):
            self.wandb.config.update(
                extra_repr_dict(obj=attack.goal_function, prefix="goal_"))


def attack_result_type_to_str(attack_result_type):
    type_name_map = {
        SuccessfulAttackResult: "SuccessfulAttackResult",
        MaximizedAttackResult: "MaximizedAttackResult",
        FailedAttackResult: "FailedAttackResult",
        SkippedAttackResult: "SkippedAttackResult",
    }
    if attack_result_type not in type_name_map:
        return str(attack_result_type)
    return type_name_map[attack_result_type]
