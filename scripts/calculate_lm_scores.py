from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import nlp
import textattack

from language_model import LanguageModelScorer


def main():
    lm = LanguageModelScorer(language_model_name='distilbert-base-uncased')
    sst2 = nlp.load_dataset('glue', 'sst2')['train']

    lm_scores = []
    num_examples = 1000
    for s in tqdm(sst2[:num_examples]['sentence']):
        lm_scores.append(lm.get_text_score(textattack.shared.AttackedText(s)))

    print(f"Stats for the first {num_examples} examples in SST:")
    print(pd.DataFrame(lm_scores).describe())


if __name__ == '__main__':
    main()
