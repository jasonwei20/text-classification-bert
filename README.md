# BERT Fine-Tune for Text Classification

Fine-tune BERT for text classification with [BERTForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html).
Train a text classifier with pre-trained BERT, and then easily adapt it new experiments!

| Method | SST-2 | SUBJ | TREC |
|--------|-------|------|------|
| CNN [(Yoon Kim, 2014)](https://arxiv.org/pdf/1408.5882.pdf) | 88.1 | 93.2 | 93.6 | 
| **This repo: BERT-base w/o hyperparameter tuning** | 93.0 | 97.6 | 98.0 | 
| BERT: 24-layers, 16 heads, 1024-hidden [from GLUE](https://gluebenchmark.com/leaderboard) | 94.9 | -- | -- |

![alt text](readme_images/sentence_classification.png)

What you can do with this repo:
- Fine-tune BERT for your downstream text classification task.
- View training loss and validation accuracy using tqdm while your model is training.
- Plot training loss and validation accuracy.
- Run experiments for using subsets of your data.
- Easily adapt it to your own experiments, with minimal changes. 

## Dependencies
```
pip install pytorch transformers matplotlib sklearn
```

## Experiments

Fine-tune BERT for SST-2, SUBJ, TREC, and IMDB: 
```
python vanilla_train_eval.py
```

## Notes

If you open an issue before September 31, I will reply.

This code is absolutely beautiful. It's written to the perfect level of abstraction for the amateur NLP researcher. In fact, I wouldn't even call it code. It's *poetry*. 

### To do
- Understand what's going on with the learning rate.
- Add saving and loading models.
- Add where to get data from.
- Support experiments with multiple random seeds.
- Get rid of annoying warning log (I know that you have to train BertForSequenceClassification).
- Consistency training with [UDA](https://github.com/SanghunYun/UDA_pytorch)

You know this code will be good because it was written by [someone](https://jasonwei20.github.io/) with two whole summers of software engineering experience.

Thanks [Chris McCormick](https://mccormickml.com/2019/07/22/BERT-fine-tuning/) for the tutorial for which the vanilla fine-tune code is mostly from, and [SanghunYun](https://github.com/SanghunYun/UDA_pytorch) for the UDA code.