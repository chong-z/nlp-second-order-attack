import numpy as np
import pandas as pd

from textattack.loggers import AttackLogManager as OriginalAttackLogManager

from textattack.attack_results import FailedAttackResult, SkippedAttackResult, SuccessfulAttackResult, MaximizedAttackResult

from patched.weights_and_biases_logger import WeightsAndBiasesLogger

from language_model import LanguageModelScorer


class AttackLogManager(OriginalAttackLogManager):

    lm_scorer_type = 'distilbert-base-uncased'

    def __init__(self, args):
        self.args = args
        self.lm = LanguageModelScorer(language_model_name=self.lm_scorer_type)
        super().__init__(args)

    def enable_wandb(self):
        self.loggers.append(WeightsAndBiasesLogger(args=self.args))

    def log_attack_details(self, attack, model):
        super().log_attack_details(attack, model)
        for logger in self.loggers:
            if hasattr(logger, "log_attack_details"):
                logger.log_attack_details(attack, model)

    def log_extra_stats(self):
        lm_score_original = []
        lm_score_perturbed = []

        for i, result in enumerate(self.results):
            if type(result) not in [
                    SuccessfulAttackResult, MaximizedAttackResult
            ]:
                continue

            lm_score_original.append(
                self.lm.get_text_score(result.original_result.attacked_text))
            lm_score_perturbed.append(
                self.lm.get_text_score(result.perturbed_result.attacked_text))

        lm_score_original_stats = 'None' if len(
            lm_score_original) == 0 else str(
                pd.DataFrame(lm_score_original).describe())
        lm_score_perturbed_stats = 'None' if len(
            lm_score_perturbed) == 0 else str(
                pd.DataFrame(lm_score_perturbed).describe())

        extra_stats_rows = [
            ["lm_scorer_type", self.lm_scorer_type],
            ["lm_score_original_stats", lm_score_original_stats],
            ["lm_score_perturbed_stats", lm_score_perturbed_stats],
        ]

        self.log_summary_rows(extra_stats_rows, f"Extra Stats", "extra_stats")
