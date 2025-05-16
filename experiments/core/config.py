import os
from pathlib import Path
from typing import NamedTuple

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

    batch_size: int = 64
    dev_batch_size: int = 256
