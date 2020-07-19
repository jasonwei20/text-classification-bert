
import json
from typing import NamedTuple

class config(NamedTuple):

    seed: int = 0
    max_length: int = 64
    batch_size: int = 32

    train_path: str = None
    test_path: str = None


    @classmethod
    def from_json(cls, file_path):
        return cls(**json.load(open(file_path, 'r')))