import random
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer, scoring
from matplotlib import pyplot as plt
import pickle
import re
import time


def set_device():
    use_GPU = torch.cuda.is_available()
    device = "cuda" if use_GPU else "cpu"
    if use_GPU:
        print("Device: " + str(device))
        print("GPU: " + str(torch.cuda.get_device_name(0)))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    else:
        print("Use GPU: " + str(use_GPU))
    return device


def set_rand_seeds():
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def calculate_rouge(output_lns, reference_lns):
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}


def saveInfo(performance):
    pickle.dump(performance, open('./Info.pkl', 'wb'))


def loadInfo():
    data = pickle.load(open('/Info.pkl', 'rb'))
    return data["train"], data["valid"], data["test"], data["test_perform"]


def plotLossWithEpochs(loss_train, loss_val, epochs):
    plt.plot(list(range(epochs)), loss_train)
    plt.plot(list(range(epochs)), loss_val)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["train_loss", "val_loss"])
    plt.show()


######################################################################################
######################################################################################

"""Learning rate scheduler"""

import math
import torch
from torch.optim.optimizer import Optimizer


class LRSchedular(object):
    """Base class for learning rate schedular
    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, last_step=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.last_step = last_step
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.lrs = self.base_lrs
        self.step()

    def state_dict(self):
        """Returns the state of the learning rate scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the learning rate scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        return self.base_lrs

    def step(self, step=None):
        """Update the learning rates.
        Arguments:
            step (int): The index of current step. (Default: None)
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        lr_values = self.get_lr()
        self.lrs = lr_values

        for group, lr in zip(self.optimizer.param_groups, lr_values):
            group["lr"] = lr


class LinearWarmupRsqrtDecayLR(LRSchedular):
    """Learning rate warmup at the beginning then decay.

    References:
        https://arxiv.org/pdf/1804.00247.pdf Section 4.6
        https://github.com/google-research/pegasus/blob/13fcf2b1191e0df950436c82c33d672c1447f5ff/pegasus/params/estimator_utils.py#L112

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_step (int): Number of step for warmup.
        last_step (int): Last step. Default: -1.

    """

    def __init__(self, optimizer, warmup_step, last_step=-1):
        self.warmup_step = warmup_step
        super(LinearWarmupRsqrtDecayLR, self).__init__(optimizer, last_step)

    def get_lr(self):
        lr_values = list()
        for base_lr in self.base_lrs:
            lr = (
                    base_lr
                    * math.sqrt(self.warmup_step)
                    / math.sqrt(max(self.last_step, self.warmup_step))
            )
            lr = min((self.last_step + 1) / self.warmup_step * base_lr, lr)
            lr_values.append(lr)
        return lr_values


######################################################################################
######################################################################################


# def map_ontology_to_tokens(tokens, ids):
#     if "[]" in ids:
#         return []

#     ids = re.findall("\d+", ids)
#     num = 0
#     idx = 0
#     target_token = []
#     for i in range(len(tokens)):
#         token = tokens[i]
#         num += len(token)
#         if "<s>" in token:
#             num -= 3
#         if "</s>" in token:
#             num -= 4
#         if token.startswith("_"):
#             num -= 1
#         if num > int(ids[idx]):
#             target_token.append(i)
#             idx += 1
#         if idx == len(ids):
#             break
#     return target_token


# def compute_length_prob(p=0.2, min_length=1, max_length=10):
#     q = 1 - p
#     length = max_length - min_length + 1

#     # Normalized
#     pb = [(p / (1 - q ** length)) * (q ** i) for i in range(min_length - 1, max_length)]

#     # average_length
#     avg_len = sum([pb[i] * (min_length + i) for i in range(length)])

#     print("Average span length: ", avg_len)

#     return pb, avg_len


# def masking(text, idx, num_of_spans):
#     text_idx = list(range(len(text)))
#     num_of_ontol = len(idx)

#     if num_of_ontol >= num_of_spans:
#         # select number of spans
#         selection = np.sort(np.random.choice(idx, size=num_of_spans, replace=False))
#     else:
#         # select all ontologies and randomly select other tokens
#         list_left = np.setdiff1d(text_idx, idx)
#         select_left = np.random.choice(list_left, size=num_of_spans - num_of_ontol, replace=False)
#         selection = np.sort(np.concatenate([np.array(idx), select_left])).astype(int)

#     return selection


# def span_masking(text, idx, span_length):
#     span = [span_length] * len(idx)

#     mask = np.ones(len(text), dtype=bool)

#     for i, k in zip(idx, span):
#         end = i + k
#         begin = i

#         if end > len(text) - 1:
#             end = len(text) - 1

#         if begin < 0:
#             begin = 0

#         # Check the fragment ends with whole word
#         # for j in range(span_length):
#         while not is_end_with_whole_word(text, end):
#             end += 1

#         # Check the fragment starts with whole word
#         while not is_begin_with_whole_word(text, begin):
#             begin -= 1

#         mask[begin: end] = False

#     return mask


# def is_end_with_whole_word(text, end):
#     # For 3 words
#     # if end < len(text)-1 and not text[end+1].startswith("▁"):
#     if end < len(text) - 1 and not text[end].startswith("▁"):
#         return False
#     return True


# def is_begin_with_whole_word(text, begin):
#     if begin > 0 and not text[begin].startswith("▁"):
#         return False
#     return True


# sentinel = "<extra_id_"


# def replace_with_sentinel(text, mask):
#     index = 0
#     in_span = False
#     fragment = []

#     for x, y in zip(text, mask):
#         if y == True:
#             fragment.append(x)
#             in_span = False
#         elif not in_span:
#             fragment.append(f"{sentinel}{index}>")
#             in_span = True
#             index += 1

#     # No bos for T5, but need eos
#     fragment.append("</s>")
#     return fragment


# def pretrained_data_preprocess(tok, csv_path, end):
#     data = pd.read_csv(csv_path)

#     if end:
#         data = data.iloc[:end]

#     encode_inputs = []
#     decode_inputs = []

#     avg_length = 3
#     mask_prob = 0.15

#     start = time.time()

#     for text, ontol in zip(data["Text"], data["Ontol"]):
#         # Tokens
#         tokens = tok.tokenize(text)

#         num_of_span = math.floor((len(tokens) * mask_prob) / avg_length)

#         ontol_idx = map_ontology_to_tokens(tokens, ontol)

#         # The tokens need to be masked
#         ontol_token_idx = masking(tokens, ontol_idx, num_of_span)
#         # span mask for each text tokens
#         mask = span_masking(tokens, ontol_token_idx, avg_length)

#         encode_inputs.append(replace_with_sentinel(tokens, mask))
#         decode_inputs.append(replace_with_sentinel(tokens, ~mask))

#     print("Processing time: {:.3f}".format(time.time() - start))

#     return encode_inputs, decode_inputs


# def train_val_split(data_path, val=False, index=1):
#     data = pd.read_csv(f"{data_path}/pretraining{index}.csv")
#     print(data.head())

#     # A txt file to sentence piece training
#     f = open(data_path + "/train_spm.txt", "a+")

#     # Shuffling
#     data = data.sample(frac=1, replace=False)

#     data[:250000].to_csv(f"{data_path}/pretrain_train{index * 2 - 1}.csv", index=False)
#     data[250000:500000].to_csv(f"{data_path}/pretrain_train{index * 2}.csv", index=False)

#     if val:
#         data[500000:].to_csv(f"{data_path}/pretrain_val.csv", index=False)

#     f.writelines(data[:500000]["Text"])

#     f.close()



