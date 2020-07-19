# BERT Fine-Tune for Text Classification, Optimized for Adaptablity

Fine-tuning BERT for text-classification with [BERTForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html) from HuggingFace.

Easily train a text classifier with pre-trained BERT, and then adapt it to your experiments!

What you can do with this repo:
- Fine-tune BERT for your downstream text classification task.
- Plot training and validation loss and accuracy.
- Run experiments for using subsets of your data.
- Run experiments with multiple random seeds.
- Easily adapt it to your own experiments, with minimal changes. 

To do:
- Consistency training with [UDA](https://github.com/SanghunYun/UDA_pytorch)

You know this code will be good because it was written by [someone](https://jasonwei20.github.io/) with two whole summers of software engineering experience.

Want more?
- Chris McCormick has a [good tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)