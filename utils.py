import os, sys
import importlib
import transformers
import pandas as pd
import json

CACHE_DIR = '.cache'


def import_module_from_path(module_name, module_path=None):
    if module_path is None:
        module_path = '/'.join(module_name.split('.')[:-1])
    path = f'{os.getcwd()}/{module_path}'
    if path not in sys.path:
        sys.path.append(path)
    return importlib.import_module(module_name)


def find_words_in_list(word_list, needles):
    ids = []
    for i in range(len(word_list)):
        if word_list[i] in needles:
            ids.append(i)
    return ids


def extra_repr_dict(obj, prefix=''):
    return dict(
        (f"{prefix}{k}", obj.__dict__[k]) for k in obj.extra_repr_keys())


def get_gender_define_words():
    url_list = [
        'https://raw.githubusercontent.com/uclanlp/gn_glove/master/wordlist/male_word_file.txt',
        'https://raw.githubusercontent.com/uclanlp/gn_glove/master/wordlist/female_word_file.txt',
    ]

    result = []
    for url in url_list:
        path = transformers.file_utils.cached_path(url)
        with open(path) as f:
            lines = f.readlines()
        new_words = [l.strip() for l in lines]
        result.extend(new_words)
    return result, url_list


def get_gender_swap_words():
    url_list = [
        'https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/generalized_swaps.txt',
        'https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/extra_gendered_words.txt',
    ]

    result = []
    for url in url_list:
        path = transformers.file_utils.cached_path(url)
        with open(path) as f:
            lines = f.readlines()
        new_words = [l.split('\t') for l in lines]
        new_words = [(l[0].strip(), l[1].strip()) for l in new_words]
        result.extend(new_words)
    return result, url_list


def get_gender_neutralize_pairs():
    url_list = [
        relative_path('small_data/gender_neutralize_list.tsv'),
    ]

    result = []
    for url in url_list:
        df = pd.read_csv(url, sep='\t')
        new_words = zip(df['from'].tolist(), df['to'].tolist())
        new_words = [(l[0].strip(), l[1].strip()) for l in new_words]
        result.extend(new_words)
    return result


def get_extra_swap_list():
    url_list = [
        relative_path('small_data/extra_swap_list.tsv'),
    ]

    result = []
    for url in url_list:
        df = pd.read_csv(url, sep='\t')
        new_words = zip(df['w1'].tolist(), df['w2'].tolist())
        new_words = [(l[0].strip(), l[1].strip()) for l in new_words]
        result.extend(new_words)
    return result, url_list


def get_anchor_words(model, gen_type='counterfitted'):
    model_name = model.name.replace(':', '_').replace('/', '_')
    gen_type_str = gen_type if type(gen_type) == str else '-'.join(gen_type)
    path_to_anchor_words = f'{CACHE_DIR}/{gen_type_str}_anchord_words_{model_name}.json'

    if os.path.exists(path_to_anchor_words):
        print(f'Loading {gen_type} anchor words from {path_to_anchor_words}')
        with open(path_to_anchor_words, 'r') as f:
            anchor_words = json.load(f)
    else:
        print(f'Generating {gen_type} anchor words to {path_to_anchor_words}')
        from scripts.gen_anchor_words import get_anchor_words_for_model
        anchor_words = get_anchor_words_for_model(model,
                                                  model.tokenizer,
                                                  gen_type=gen_type)
        with open(path_to_anchor_words, 'w') as outfile:
            json.dump(anchor_words, outfile)

    anchor_words = [(p[0], p[1]) for p in anchor_words]
    return anchor_words


# Requires |attack_attrs['active_biaswords']|.
def get_active_biaswords(attacked_text):
    active_biaswords = get_attack_attr(attacked_text, 'active_biaswords')
    assert active_biaswords is not None, 'the initial attacked text must have `active_biaswords`'
    return active_biaswords


def get_attack_attr(attacked_text, key):
    t = attacked_text
    while key not in t.attack_attrs and 'previous_attacked_text' in t.attack_attrs:
        t = t.attack_attrs['previous_attacked_text']
    if key not in t.attack_attrs:
        return None
    attacked_text.attack_attrs[key] = t.attack_attrs[key]
    return attacked_text.attack_attrs[key]


def generate_biased_texts(input_text):
    active_biaswords = get_active_biaswords(input_text)
    ids = find_words_in_list(input_text.words, active_biaswords)
    return [
        input_text.replace_words_at_indices(ids, [w] * len(ids))
        for w in active_biaswords
    ]


def replace_word(input_text, old_word, new_word):
    ids = find_words_in_list(input_text.words, [old_word])
    return input_text.replace_words_at_indices(ids, [new_word] * len(ids))


def biaswords_list_from_str(biaswords_str):
    if biaswords_str is None:
        return None
    return [tuple(biaswords_str.split(","))]


def relative_path(path):
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_directory, path)


def flatten_nested_list(nested_list):
    flatten = set()
    for words in nested_list:
        for w in words:
            flatten.add(w)
    return list(flatten)
