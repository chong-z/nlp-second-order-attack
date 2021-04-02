import numpy as np
import random
import math
import torch

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import BeamSearch


def unique_attacked_texts(all_attacked_texts):
    h = set()
    result_attacked_texts = []
    for attacked_text in all_attacked_texts:
        if attacked_text.text in h:
            continue
        h.add(attacked_text.text)
        result_attacked_texts.append(attacked_text)
    return result_attacked_texts


def format_data_for_logging(d):
    if isinstance(d, int):
        return d
    if isinstance(d, float):
        return round(d, 4)
    if isinstance(d, list):
        return [format_data_for_logging(v) for v in d]
    if isinstance(d, dict):
        return {k: format_data_for_logging(v) for k, v in d.items()}
    if isinstance(d, torch.Tensor):
        return format_data_for_logging(d.tolist())
    return d


class BeamSearchPlus(BeamSearch):

    def __init__(self,
                 beam_width=8,
                 beam_sampling_method='max',
                 transformations_sampling_ratio=1,
                 search_depth=100):
        self.beam_width = beam_width
        self.beam_sampling_method = beam_sampling_method
        self.transformations_sampling_ratio = transformations_sampling_ratio
        self.search_depth = search_depth

    def _perform_search(self, initial_result):
        self.intermediate_results_for_logging = []
        best_result = self._perform_search_internal(initial_result)
        best_result.extra_logging = {
            "intermediate_results_for_logging":
                format_data_for_logging(self.intermediate_results_for_logging),
        }
        return best_result

    def _perform_search_internal(self, initial_result):
        visited_texts = set()

        beam = [initial_result.attacked_text]
        depth = 0
        best_result = initial_result

        self.intermediate_results_for_logging.append({
            "depth": 0,
            "perturbed_outputs": [initial_result.raw_output.tolist()],
        })

        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED and depth < self.search_depth:
            depth += 1
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text)
                if self.transformations_sampling_ratio < 1:
                    transformations = random.sample(
                        transformations,
                        math.ceil(
                            len(transformations) *
                            self.transformations_sampling_ratio))
                for next_text in transformations:
                    potential_next_beam.append(next_text)
            potential_next_beam = unique_attacked_texts(potential_next_beam)
            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return best_result
            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])

            if self.beam_sampling_method == 'max':
                tmp_best_result = results[scores.argmax()]
                if tmp_best_result.score > best_result.score:
                    best_result = tmp_best_result
            elif self.beam_sampling_method == 'random':
                best_result = random.choice(results)
            else:
                raise ValueError(
                    f"Unsupported beam_sampling_method:{self.beam_sampling_method}."
                )

            print(
                f"best_score: {best_result.score} raw_output: {best_result.raw_output} text: {best_result.attacked_text.text}"
            )

            if best_result.goal_status == GoalFunctionResultStatus.MAXIMIZING:
                # Only log detailed results for analyzing tasks
                self.intermediate_results_for_logging.append({
                    "depth":
                        depth,
                    "perturbed_outputs": [
                        result.raw_output.tolist() for result in results
                    ],
                    "num_diff_words": [
                        len(result.attacked_text.
                            attack_attrs["modified_indices"])
                        for result in results
                    ],
                })

            if search_over:
                return best_result

            active_beam_width = self.beam_width

            if self.beam_sampling_method == 'max':
                # Refill the beam with best results.
                sorted_scores = (-scores).argsort()
            elif self.beam_sampling_method == 'random':
                active_beam_width = self.beam_width
                # Refill the beam with random sampling.
                idx = random.sample(
                    range(len(potential_next_beam)),
                    min(active_beam_width, len(potential_next_beam)))
                potential_next_beam = [potential_next_beam[i] for i in idx]
                results = [results[i] for i in idx]
                scores = scores[idx]
                sorted_scores = list(range(len(scores)))
            else:
                raise ValueError(
                    f"Unsupported beam_sampling_method:{self.beam_sampling_method}."
                )

            best_indices = []
            for i in sorted_scores:
                if potential_next_beam[i].text not in visited_texts:
                    visited_texts.add(potential_next_beam[i].text)
                    best_indices.append(i)
                    if len(best_indices) >= active_beam_width:
                        break

            # best_indices = (-scores).argsort()[:self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]

            print(f"  depth = {depth}, len(beam) = {len(beam)}")
            for i in best_indices[:5]:
                print(
                    f"  beam_succeed: {results[i].goal_status == GoalFunctionResultStatus.SUCCEEDED} score: {results[i].score} raw_output: {results[i].raw_output} text: '{results[i].attacked_text.text}'"
                )

        return best_result

    def extra_repr_keys(self):
        return [
            "beam_width", "beam_sampling_method",
            "transformations_sampling_ratio", "search_depth"
        ]
