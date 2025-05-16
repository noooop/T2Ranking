import os
from pathlib import Path
from typing import NamedTuple

import torch

t2ranking_data = Path(os.environ.get("T2RANKING_DATA", ""))

assert t2ranking_data.exists(), "T2RANKING_DATA not found!"


class CrossEncoderConfig(NamedTuple):
    model_name_or_path: str = "google-bert/bert-base-chinese"

    collection: str = "collection.tsv"

    query: str = "queries.train.tsv"
    top1000: str = "train.bm25.tsv"
    qrels: str = "qrels.retrieval.train.tsv"

    dev_query: str = "queries.dev.tsv"
    dev_top1000: str = "dev.bm25.tsv"
    dev_qrels: str = "qrels.retrieval.dev.tsv"

    max_seq_len: int = 332
    min_index: int = 0
    max_index: int = 256
    sample_num: int = 64

    epochs: int = 4

    batch_size: int = 8
    dev_batch_size: int = 512

    learning_rate: float = 2e-5

    model_out_dir: str = "output"

    dtype: str = "float16"
    report: int = 100
    gradient_checkpoint: bool = True

    t2ranking_data: Path = t2ranking_data

    @property
    def torch_dtype(self):
        if self.dtype == "bfloat16":
            return torch.bfloat16
        elif self.dtype == "float16":
            return torch.float16
        elif self.dtype == "float32":
            return torch.float32
        else:
            raise ValueError("dtype does not supported.")
