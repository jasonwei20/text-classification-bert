import json
from typing import List, NamedTuple

class config(NamedTuple):

    exp_id: str = None
    seed_num: int = 0

    #dataset params
    train_path: str = None
    test_path: str = None
    num_output_classes: int = None

    # model params
    max_length: int = 64

    # training params
    train_subset: int = None
    val_subset: int = None
    batch_size: int = 32
    num_epochs: int = 4
    eval_interval: int = 10

    @classmethod
    def from_json(cls, file_path):
        return cls(**json.load(open(file_path, 'r')))

class uda_config(NamedTuple):

    exp_id: str = None
    seed_num: int = 0

    #dataset params
    train_path: str = None
    test_path: str = None
    num_output_classes: int = None

    # model params
    max_length: int = 64

    # training params
    train_subset: int = None
    val_subset: int = 1000
    total_updates: int = 10000
    eval_interval: int = 10

    # uda
    uda_mode: str = None
    uda_train_batch_size: int = None
    uda_ul_batch_size: int = None
    val_batch_size: int = 64
    uda_augmentation: str = None #"token_swaps", "backtranslation", "synonym_replacement"

    @classmethod
    def from_json(cls, file_path):
        return cls(**json.load(open(file_path, 'r')))