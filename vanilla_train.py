from transformers import BertForSequenceClassification, AdamW, BertConfig
from utils import common

#load the dataset
def get_dataloaders():
    return NotImplemented

#train the model
def fine_tune_bert(

):
    model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    model.cuda()

if __name__ == "__main__":
    print("hello")
    common.print_jason()