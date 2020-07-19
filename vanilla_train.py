import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig, AdamW
from utils import common, configuration
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# load the dataset
def get_dataloader(
    path, 
    cfg,
    ):
    
    sentences, labels = common.get_sentences_and_labels_from_txt(path)
    input_id_list = []; attention_mask_list = []

    for sentence in sentences[:96]:

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
    labels_torch = torch.tensor(labels[:96])

    dataset = TensorDataset(input_ids_torch, attention_masks_torch, labels_torch)

    dataloader = DataLoader(
        dataset, 
        sampler = RandomSampler(dataset), 
        batch_size = cfg.batch_size,
    )

    return dataloader

# train the model
def finetune_bert(
    train_dataloader,
    ):

    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))

    for mb_num, mb in enumerate(train_dataloader):
        input_ids = mb[0].to(device)
        input_mask = mb[1].to(device)
        labels = mb[2].to(device)

        print(mb_num, input_ids.shape, input_mask.shape, labels.shape)
        
#     model = BertForSequenceClassification.from_pretrained(
#         "bert-base-uncased",
#         num_labels = 2, # The number of output labels--2 for binary classification.
#                         # You can increase this for multi-class tasks.   
#         output_attentions = False, # Whether the model returns attentions weights.
#         output_hidden_states = False, # Whether the model returns all hidden-states.
#     )

#     model.cuda()

if __name__ == "__main__":

    cfg = configuration.config.from_json("config/sst2.json")
    train_dataloader = get_dataloader(cfg.train_path, cfg)
    finetune_bert(train_dataloader)