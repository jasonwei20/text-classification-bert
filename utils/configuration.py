import json
from typing import List, NamedTuple

class config(NamedTuple):

    seed_num: int = 0

    #dataset params
    train_path: str = None
    test_path: str = None
    num_output_classes: int = None

    # model params
    max_length: int = 64

    # training params
    train_subset: int = None
    batch_size: int = 32
    num_epochs: int = 4

    @classmethod
    def from_json(cls, file_path):
        return cls(**json.load(open(file_path, 'r')))