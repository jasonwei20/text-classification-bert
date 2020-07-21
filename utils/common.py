import random
import numpy as np
import torch

def set_random_seed(seed_num):

    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

def get_sentences_and_labels_from_txt(dataset_path):
    lines = open(dataset_path).readlines()
    sentences = []
    labels = []
    for line in lines:
        parts = line[:-1].split('\t')
        label = int(parts[0])
        sentence = parts[1]
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels

def repeat_dataloader(iterable):
    while True:
        for x in iterable:
            yield x