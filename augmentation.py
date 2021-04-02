import utils

from textattack.augmentation import Augmenter
from textattack.shared import AttackedText
from textattack.transformations import Transformation

from scripts.gen_anchor_words import load_neighbors


class CounterfactualSwap(Transformation):

    def __init__(self, swap_types):
        self.replace_dict = {}
        self.swap_types = swap_types

        neighbors = []
        for swap_type in self.swap_types:
            neighbors.extend(list(load_neighbors(swap_type).items()))

        self.sample_swap_list = neighbors[:5] + neighbors[-5:]

        for w1, w2s in neighbors:
            for w2 in w2s:
                # Only keep the first swap.
                if w1 not in self.replace_dict:
                    self.replace_dict[w1] = w2
                if w2 not in self.replace_dict:
                    self.replace_dict[w2] = w1

    # Returns |2^n| transformed texts, where |n| is the number of words in |self.replace_dict|.
    def _get_transformations(self, current_text, indices_to_modify=None):
        words = current_text.words
        transformed_texts = [current_text]

        for i in range(len(words)):
            if words[i] in self.replace_dict:
                # Double the size of |transformed_texts|.
                new_word = self.replace_dict[words[i]]
                old_len = len(transformed_texts)
                for j in range(old_len):
                    transformed_texts.append(
                        transformed_texts[j].replace_word_at_index(i, new_word))

        return transformed_texts

    def extra_repr_keys(self):
        return ["swap_types", "sample_swap_list"] + super().extra_repr_keys()


class GenderAugmenter(Augmenter):

    def __init__(
        self,
        pct_words_to_swap=0.1,
        transformations_per_example=1,
        swap_types=['gender'],
    ):
        self.transformation = CounterfactualSwap(swap_types=swap_types)
        self.constraints = []
        self.pre_transformation_constraints = []

    def augment(self, text):
        attacked_text = AttackedText(text)
        transformed_texts = self.transformation._get_transformations(
            attacked_text)
        return [t.printable_text() for t in transformed_texts]

    def augment_with_label(self, text, label):
        texts = self.augment(text)
        return zip(texts, [label] * len(texts))


class BiasHider(Augmenter):
    # Swap SST-2 dev and append to the first example.

    def __init__(
        self,
        pct_words_to_swap=0.1,
        transformations_per_example=1,
    ):
        self.transformation = {
            'he': 'she',
            'she': 'he',
            'gay': 'straight',
            'straight': 'gay',
        }
        self.constraints = []
        self.pre_transformation_constraints = []

        self.should_append_swap = True

    def augment(self, text):
        raise NotImplementedError()

    def augment_with_label(self, text, label):
        if not self.should_append_swap:
            return [(text, label)]

        self.should_append_swap = False
        swapped_dev = list(self.get_swapped_dev())
        swapped_dev.append((text, label))
        return swapped_dev

    def get_swapped_dev(self):
        import nlp
        dev = nlp.load_dataset('glue', 'sst2')['validation']

        swapped_texts = []
        swapped_lables = []

        trans = CounterfactualSwap(swap_types=['gender'])
        trans.replace_dict = self.transformation

        for e in dev:
            text = e['sentence']
            label = e['label']
            attacked_text = AttackedText(text)
            transformed_texts = trans._get_transformations(attacked_text)
            if len(transformed_texts) == 1:
                # Does not contain patch words.
                continue
            swapped_texts.extend(
                [t.printable_text() for t in transformed_texts])
            swapped_lables.extend([label] * len(transformed_texts))

        return zip(swapped_texts, swapped_lables)
