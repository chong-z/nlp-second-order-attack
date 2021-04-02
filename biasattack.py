import numpy as np
import random
import torch

import textattack
from textattack.shared.attack import Attack
from textattack.transformations import CompositeTransformation, WordDeletion

import utils
from constraints import WordBlacklistConstraint, ActiveAnchorWordsModification, LanguageModelScoreConstraint
from goal_function import BiasGoalFunctionResult, BiasGoalFunction
from search_method import BeamSearchPlus
from transformations import Identity, InitialBiasWord, WordSwapMaskedLMPlus


def BiasAttack(model,
               score_mode,
               biaswords_list=None,
               biaswords_type='counterfitted',
               biasthreshold=0.5,
               diffthreshold=0.03,
               max_masks=1,
               max_trials=-1,
               beam_width=5,
               max_candidates=20,
               logit_threshold=3,
               beam_sampling_method='max',
               transformations_sampling_ratio=1,
               search_depth=100,
               query_budget=5000,
               maximizable=False):
    """
        BiasAttack main entrance.
    """
    np.set_printoptions(precision=6)

    if biaswords_list is None:
        # biaswords_list, bias_urls = utils.get_gender_swap_words()
        biaswords_list = utils.get_anchor_words(model, gen_type=biaswords_type)
        print(
            f"Loaded {len(biaswords_list)} biaswords from utils.\n Samples: {biaswords_list[:10]}...{biaswords_list[-10:]}"
        )
    else:
        print(
            f"Loaded {len(biaswords_list)} biaswords from command line args.\n Samples: {biaswords_list[:10]}"
        )

    biaswords_flatten = utils.flatten_nested_list(biaswords_list)

    transformation = CompositeTransformation([
        InitialBiasWord(biaswords_flatten=biaswords_flatten),
        WordSwapMaskedLMPlus(max_masks=max_masks,
                             max_trials=max_trials,
                             max_candidates=max_candidates,
                             logit_threshold=logit_threshold,
                             max_length=128,
                             masked_language_model="distilbert-base-uncased"),
    ])

    blacklist_words = [*biaswords_flatten]
    gender_define_words, gender_define_urls = utils.get_gender_define_words()
    blacklist_words.extend(gender_define_words)
    blacklist_words = [w.strip() for w in blacklist_words]
    print(
        f"Loaded {len(blacklist_words)} blacklist_words from {gender_define_urls}.\n Samples: {blacklist_words[:10]}...{blacklist_words[-10:]}"
    )

    constraints = [
        ActiveAnchorWordsModification(),
    ]

    # Note: Each |query_budget| corresponds to |len(biaswords)| actual model queries.
    gpu_memory_MB = torch.cuda.get_device_properties(
        textattack.shared.utils.device).total_memory // (2**20)
    print(f'gpu_memory_MB = {gpu_memory_MB}')
    if gpu_memory_MB >= 8000:
        model_batch_size = 512
    else:
        model_batch_size = 128

    goal_function = BiasGoalFunction(model,
                                     maximizable=maximizable,
                                     biaswords_list=biaswords_list,
                                     active_biaswords_logit_threshold=1.5,
                                     biasthreshold=biasthreshold,
                                     diffthreshold=diffthreshold,
                                     stepweight=0.1,
                                     score_mode=score_mode,
                                     lm_scorer="distilbert-base-uncased",
                                     query_budget=query_budget,
                                     model_batch_size=model_batch_size)

    search_method = BeamSearchPlus(
        beam_width=beam_width,
        beam_sampling_method=beam_sampling_method,
        transformations_sampling_ratio=transformations_sampling_ratio,
        search_depth=search_depth)

    return Attack(goal_function, constraints, transformation, search_method)


def SOEnumAttack(model, biaswords_str):
    return BiasAttack(
        model,
        score_mode="max",
        biaswords_type='counterfitted',
        biaswords_list=utils.biaswords_list_from_str(biaswords_str),
        beam_width=2000,
        search_depth=2,
        query_budget=100000)


def SOBeamAttack(model, biaswords_str):
    return BiasAttack(
        model,
        score_mode="max",
        biaswords_type='counterfitted',
        biaswords_list=utils.biaswords_list_from_str(biaswords_str),
        max_masks=1,
        max_trials=-1,
        beam_width=20,
        search_depth=6,
        query_budget=50000)


def RandomBaselineAttack(model, biaswords_str):
    return BiasAttack(
        model,
        score_mode="max",
        beam_sampling_method="random",
        biaswords_type='counterfitted',
        biaswords_list=utils.biaswords_list_from_str(biaswords_str),
        max_masks=1,
        max_trials=1,
        beam_width=1,
        search_depth=6,
        query_budget=100)


def RandomNoLMBaselineAttack(model, biaswords_str):
    return BiasAttack(
        model,
        score_mode="max",
        beam_sampling_method="random",
        biaswords_type='counterfitted',
        biaswords_list=utils.biaswords_list_from_str(biaswords_str),
        max_candidates=1000,
        logit_threshold=10,
        max_masks=1,
        max_trials=1,
        beam_width=1,
        search_depth=6,
        query_budget=100)


def BiasAnalysisChecklist(model, biaswords_str):
    return BiasAttack(
        model,
        score_mode="max",
        biaswords_type='checklist',
        biaswords_list=utils.biaswords_list_from_str(biaswords_str),
        beam_sampling_method='random',
        beam_width=800,
        search_depth=3,
        query_budget=100000,
        maximizable=True)


def BiasAnalysisGender(model, biaswords_str):
    return BiasAttack(
        model,
        score_mode="max",
        biaswords_type='gender',
        biaswords_list=utils.biaswords_list_from_str(biaswords_str),
        beam_sampling_method='random',
        beam_width=800,
        search_depth=3,
        query_budget=100000,
        maximizable=True)
