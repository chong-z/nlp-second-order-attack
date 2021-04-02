import json
from tqdm.auto import tqdm
import numpy as np
import torch
import os

from collections import OrderedDict

import textattack

package_directory = os.path.dirname(os.path.abspath(__file__))

COUNTERFITTED_NEIGHBOR_FILE = os.path.join(
    package_directory, '../small_data/modified_counterfitted_neighbors.json')
CHECKLIST_NEIGHBOR_FILE = os.path.join(package_directory,
                                       '../small_data/checklist_lexicons.json')
GENDER_NEIGHBOR_FILE = os.path.join(package_directory,
                                    '../small_data/wino_male_female.json')
OUTPUT_FILE = os.path.join(package_directory,
                           '../small_data/counterfitted_anchord_words.json')


def batch_pred(texts, model, tokenizer, batch_size=128):
    attacked_texts = [textattack.shared.AttackedText(text) for text in texts]
    inputs = textattack.shared.utils.batch_tokenize(tokenizer, attacked_texts,
                                                    batch_size)
    with torch.no_grad():
        outputs = textattack.shared.utils.batch_model_predict(
            model, inputs, batch_size)
    return torch.nn.functional.softmax(outputs, dim=1)


def load_neighbors(gen_type):
    if type(gen_type) == list:
        neighbors = {}
        for t in gen_type:
            neighbors.update(load_neighbors(t))
        return neighbors

    if gen_type == 'counterfitted':
        with open(COUNTERFITTED_NEIGHBOR_FILE, 'r') as f:
            neighbors = json.load(f)
        return neighbors

    if gen_type == 'gender':
        with open(GENDER_NEIGHBOR_FILE, 'r') as f:
            male_females = json.load(f)
        neighbors = {ma: [fe] for ma, fe in male_females}
        return neighbors

    assert gen_type == 'checklist'
    with open(CHECKLIST_NEIGHBOR_FILE, 'r') as f:
        groups = json.load(f)

    allowed_groups = [
        'sexual_adj',
        'religion_adj',
        'religion',
        'nationality',
        'country',
        'male',
        'female',
        'city',
    ]
    neighbors = OrderedDict()
    for group in allowed_groups:
        tokens = groups[group]
        # We only allow single-token substitution.
        tokens = [t.lower() for t in tokens if ' ' not in t]
        for t in tokens:
            neighbors[t] = tokens
    return neighbors


# gen_type can be a str of list of strs.
def get_anchor_words_for_model(model, tokenizer, gen_type):
    neighbors = load_neighbors(gen_type)

    preds = {}
    for w1 in tqdm(neighbors):
        preds[w1] = None
        for w2 in neighbors[w1]:
            preds[w2] = None

    outputs = batch_pred(preds.keys(), model, tokenizer)
    for x, y in zip(preds.keys(), outputs.cpu()):
        preds[x] = y[1].item()

    anchord_words = []
    for w1 in tqdm(neighbors):
        p1 = preds[w1]
        for w2 in neighbors[w1]:
            diff = abs(p1 - preds[w2])
            anchord_words.append((w1, w2, diff))

    # 'checklist' and 'gender' are used for enumeration analysis.
    if gen_type != 'gender' and gen_type != 'checklist':
        anchord_words = sorted(anchord_words, key=lambda p: -p[2])
    return anchord_words


def get_anchor_words_jia(model_name='model_data/cnn_cert', model_type='cnn'):
    from models.jia_certified import ModelWrapper, vocab, word_mat, device, tokenizer, get_jia_args
    model = ModelWrapper(OPTS=get_jia_args(model_name, model_type),
                         vocab=vocab,
                         word_mat=word_mat).to(device)
    return get_anchor_words_for_model(model, tokenizer)


if __name__ == '__main__':
    anchord_words = get_anchor_words_jia()
    with open(OUTPUT_FILE, 'w') as outfile:
        json.dump(anchord_words, outfile)
