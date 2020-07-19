
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from sklearn.utils import shuffle
from utils import common
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def get_train_dataloader(cfg):
    
    sentences, labels = common.get_sentences_and_labels_from_txt(cfg.train_path)
    sentences, labels = shuffle(sentences, labels, random_state = cfg.seed_num)
    if cfg.train_subset:
        sentences = sentences[:cfg.train_subset]
        labels = labels[:cfg.train_subset]
    input_id_list = []; attention_mask_list = []

    for sentence in sentences:

        encoding_dictionary = tokenizer(
            sentence, 
            max_length = cfg.max_length,
            padding = 'max_length',
            return_tensors = 'pt',
        )

        input_ids = encoding_dictionary['input_ids']
        attention_mask = encoding_dictionary['attention_mask']
        input_id_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    input_ids_torch = torch.cat(input_id_list, dim=0)
    attention_masks_torch = torch.cat(attention_mask_list, dim=0)
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

    for sentence in sentences:

        encoding_dictionary = tokenizer(
            sentence, 
            max_length = cfg.max_length,
            padding = 'max_length',
            return_tensors = 'pt',
        )

        input_ids = encoding_dictionary['input_ids']
        attention_mask = encoding_dictionary['attention_mask']
        input_id_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    input_ids_torch = torch.cat(input_id_list, dim=0)
    attention_masks_torch = torch.cat(attention_mask_list, dim=0)
    labels_torch = torch.tensor(labels)

    dataset = TensorDataset(input_ids_torch, attention_masks_torch, labels_torch)

    dataloader = DataLoader(
        dataset, 
        sampler = SequentialSampler(dataset), 
        batch_size = cfg.batch_size,
    )

    return dataloader