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

##############################
#### Vanilla Augmentation ####
##############################

def get_augmented_train_dataloader(cfg):

    sentences, labels = common.get_sentences_and_labels_from_txt(cfg.train_path)
    sentences, labels = shuffle(sentences, labels, random_state = cfg.seed_num)
    if cfg.train_subset:
        sentences, _, labels, _ = train_test_split(sentences, labels, train_size = cfg.train_subset, random_state=cfg.seed_num, stratify=labels) 
    
    swap_sentences, backtrans_sentences, sr_sentences = augmentation.get_augmented_sentences(sentences, cfg, "all_augmentation")
    all_labels = labels + labels
    all_sentences = sentences + swap_sentences #sentences + swap_sentences + backtrans_sentences + sr_sentences
    augmentation_labels = [0 for _ in range(len(labels))] + [1 for _ in range(len(labels))]# + [2 for _ in range(len(labels))] + [3 for _ in range(len(labels))]
    
    encoding_dictionary = tokenizer(
        all_sentences, 
        max_length = cfg.max_length,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt',
    )

    input_ids_torch = encoding_dictionary['input_ids']
    attention_masks_torch = encoding_dictionary['attention_mask']
    labels_torch = torch.tensor(all_labels)
    aug_labels_torch = torch.tensor(augmentation_labels)

    dataset = TensorDataset(input_ids_torch, attention_masks_torch, labels_torch, aug_labels_torch)
    dataloader = DataLoader(
        dataset, 
        sampler = RandomSampler(dataset), 
        batch_size = cfg.batch_size,
    )

    return dataloader

#############################
############ UDA ############
#############################

def load_preloaded_data():

    def convert_part(part):
        return [int(x) for x in part[1:-1].split(', ')]

    print("using pre-loaded data")

    data_path = "data/imdb/orig_imdb_unsup_train.txt"
    lines = open(data_path, 'r').readlines()

    orig_input_ids_list = []; orig_attention_masks_list = []; aug_input_ids_list = []; aug_attention_masks_list = []
    orig_strings_list = []

    for line in lines[1:]:
        parts = line[:-1].split('\t')
        orig_input_ids_list.append(convert_part(parts[0]))
        orig_attention_masks_list.append(convert_part(parts[1]))
        aug_input_ids_list.append(convert_part(parts[3]))
        aug_attention_masks_list.append(convert_part(parts[4]))
        orig_strings_list.append(parts[0])

    orig_input_ids_torch = torch.tensor(orig_input_ids_list)
    orig_attention_masks_torch = torch.tensor(orig_attention_masks_list)
    aug_input_ids_torch = torch.tensor(aug_input_ids_list)
    aug_attention_masks_torch = torch.tensor(aug_attention_masks_list)

    return orig_input_ids_torch, orig_attention_masks_torch, aug_input_ids_torch, aug_attention_masks_torch
        
def get_tensor_uda_dataset(ul_orig_sentences, ul_aug_sentences, cfg):

    if cfg.preloaded_data == "true":
        orig_input_ids_torch, orig_attention_masks_torch, aug_input_ids_torch, aug_attention_masks_torch = load_preloaded_data()
    
    else:
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
    ul_aug_sentences = augmentation.get_augmented_sentences(ul_orig_sentences, cfg, augmentation_type=cfg.uda_augmentation)

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









##############################
######### temp stuff #########
##############################

def export_uda_dataset(cfg):
    
    sentences, labels = common.get_sentences_and_labels_from_txt(cfg.train_path)
    sentences, labels = shuffle(sentences, labels, random_state = cfg.seed_num)
    train_sentences, ul_orig_sentences, train_labels, _ = train_test_split(sentences, labels, train_size = cfg.train_subset, random_state=cfg.seed_num, stratify=labels) 
    ul_aug_sentences = augmentation.get_augmented_sentences(ul_orig_sentences, cfg, augmentation_type=cfg.uda_augmentation)

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

    output_writer = open("temp/imdb_unsup_jason_sr_train.txt", 'w')
#     output_writer.write('\t'.join(['ori_input_ids', 'ori_input_mask	ori_input_type_ids	aug_input_ids	aug_input_mask	aug_input_type_ids ']))

    for orig_input_id, orig_attention_masks, aug_input_ids, aug_attention_masks in zip(orig_input_ids_torch.tolist(), orig_attention_masks_torch.tolist(), aug_input_ids_torch.tolist(), aug_attention_masks_torch.tolist()):
        line = '\t'.join([str(orig_input_id), str(orig_attention_masks), str([0 for _ in range(128)]), str(aug_input_ids), str(aug_attention_masks), str([0 for _ in range(128)])])
        output_writer.write(line + '\n')