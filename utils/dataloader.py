import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils import common
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def get_train_dataloader(cfg):
    
    sentences, labels = common.get_sentences_and_labels_from_txt(cfg.train_path)
    sentences, labels = shuffle(sentences, labels, random_state = cfg.seed_num)
    if cfg.train_subset:
        sentences, _, labels, _ = train_test_split(sentences, labels, train_size = cfg.train_subset, random_state=cfg.seed_num, stratify=labels) 

    encoding_dictionary = tokenizer(
        sentences, 
        max_length = cfg.max_length,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt',
    )

    input_ids_torch = encoding_dictionary['input_ids']
    attention_masks_torch = encoding_dictionary['attention_mask']
    labels_torch = torch.tensor(labels)

    dataset = TensorDataset(input_ids_torch, attention_masks_torch, labels_torch)

    dataloader = DataLoader(
        dataset, 
        sampler = RandomSampler(dataset), 
        batch_size = cfg.batch_size,
    )

    return dataloader

def get_test_dataloader(cfg):
    
    sentences, labels = common.get_sentences_and_labels_from_txt(cfg.test_path)
    input_id_list = []; attention_mask_list = []

    encoding_dictionary = tokenizer(
        sentences, 
        max_length = cfg.max_length,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt',
    )

    input_ids_torch = encoding_dictionary['input_ids']
    attention_masks_torch = encoding_dictionary['attention_mask']
    labels_torch = torch.tensor(labels)

    dataset = TensorDataset(input_ids_torch, attention_masks_torch, labels_torch)

    dataloader = DataLoader(
        dataset, 
        sampler = SequentialSampler(dataset), 
        batch_size = cfg.batch_size,
    )

    return dataloader