import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from utils import augmentation, common, configuration
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def get_tensor_dataset(sentences, labels, cfg):

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

    return dataset

def get_train_dataloader(cfg):
    
    sentences, labels = common.get_sentences_and_labels_from_txt(cfg.train_path)
    sentences, labels = shuffle(sentences, labels, random_state = cfg.seed_num)
    if cfg.train_subset:
        sentences, _, labels, _ = train_test_split(sentences, labels, train_size = cfg.train_subset, random_state=cfg.seed_num, stratify=labels) 

    dataset = get_tensor_dataset(sentences, labels, cfg)
    dataloader = DataLoader(
        dataset, 
        sampler = RandomSampler(dataset), 
        batch_size = cfg.batch_size,
    )

    return dataloader

def get_test_dataloader(cfg):
    
    sentences, labels = common.get_sentences_and_labels_from_txt(cfg.test_path)
    sentences, labels = shuffle(sentences, labels, random_state = cfg.seed_num)
    if cfg.val_subset:
        sentences, _, labels, _ = train_test_split(sentences, labels, train_size = cfg.val_subset, random_state=cfg.seed_num, stratify=labels) 

    dataset = get_tensor_dataset(sentences, labels, cfg)
    dataloader = DataLoader(
        dataset, 
        sampler = SequentialSampler(dataset), 
        batch_size = cfg.val_batch_size if type(cfg) == configuration.uda_config else cfg.batch_size,
    )

    return dataloader

#############################
############ UDA ############
#############################

def get_tensor_uda_dataset(ul_orig_sentences, ul_aug_sentences, cfg):

    orig_encoding_dictionary = tokenizer(
        ul_orig_sentences, 
        max_length = cfg.max_length,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt',
    )
    orig_input_ids_torch = orig_encoding_dictionary['input_ids']
    orig_attention_masks_torch = orig_encoding_dictionary['attention_mask']

    aug_encoding_dictionary = tokenizer(
        ul_aug_sentences, 
        max_length = cfg.max_length,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt',
    )
    aug_input_ids_torch = aug_encoding_dictionary['input_ids']
    aug_attention_masks_torch = aug_encoding_dictionary['attention_mask']

    dataset = TensorDataset(orig_input_ids_torch, orig_attention_masks_torch, aug_input_ids_torch, aug_attention_masks_torch)

    return dataset

def get_train_uda_dataloader(cfg):
    
    sentences, labels = common.get_sentences_and_labels_from_txt(cfg.train_path)
    sentences, labels = shuffle(sentences, labels, random_state = cfg.seed_num)
    train_sentences, ul_orig_sentences, train_labels, _ = train_test_split(sentences, labels, train_size = cfg.train_subset, random_state=cfg.seed_num, stratify=labels) 
    ul_aug_sentences = augmentation.get_augmented_sentences(ul_orig_sentences, cfg)

    train_dataset = get_tensor_dataset(train_sentences, train_labels, cfg)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler = RandomSampler(train_dataset), 
        batch_size = cfg.uda_train_batch_size,
    )

    ul_dataset = get_tensor_uda_dataset(ul_orig_sentences, ul_aug_sentences, cfg)
    ul_dataloader = DataLoader(
        ul_dataset, 
        sampler = RandomSampler(ul_dataset), 
        batch_size = cfg.uda_ul_batch_size,
    )

    return train_dataloader, ul_dataloader