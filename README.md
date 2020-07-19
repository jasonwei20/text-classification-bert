# BERT Fine-Tune for Text Classification, Optimized for Adaptablity

Fine-tuning BERT for text-classification with [BERTForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html) from HuggingFace.

Easily train a text classifier with pre-trained BERT, and then adapt it to your experiments!

What you can do with this repo:
- Fine-tune BERT for your downstream text classification task.
- Plot training loss and validation accuracy.
- Run experiments for using subsets of your data.
- Easily adapt it to your own experiments, with minimal changes. 

This code is beautiful. It's written to the perfect level of abstraction for the amateur NLP researcher. In fact, I wouldn't even call it code. It's *poetry*. 

To do:
- Get rid of annoying error log.
- Run experiments with multiple random seeds.
- Consistency training with [UDA](https://github.com/SanghunYun/UDA_pytorch)

You know this code will be good because it was written by [someone](https://jasonwei20.github.io/) with two whole summers of software engineering experience.

Thanks [Chris McCormick](https://mccormickml.com/2019/07/22/BERT-fine-tuning/) for the tutorial for which this code is mostly from. 