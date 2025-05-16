import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .config import CrossEncoderConfig, t2ranking_data


class CrossEncoderTrainDataset(Dataset):

    def __init__(self, config: CrossEncoderConfig):
        self.min_index = config.min_index
        self.max_index = config.max_index
        self.max_seq_len = config.max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path)

        self.collection = pd.read_csv(t2ranking_data / config.collection,
                                      sep="\t",
                                      quoting=3)
        self.collection.columns = ['pid', 'para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid
        self.collection.pop('pid')

        self.query = pd.read_csv(t2ranking_data / config.query, sep="\t")
        self.query.columns = ['qid', 'text']
        self.query.index = self.query.qid
        self.query.pop('qid')

        self.top1000 = pd.read_csv(t2ranking_data / config.top1000, sep="\t")
        self.top1000 = list(self.top1000.groupby("qid"))
        self.len = len(self.top1000)

        qrels = {}
        with open(t2ranking_data / config.qrels, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                qid, pid = line.split()
                qid = int(qid)
                pid = int(pid)
                x = qrels.get(qid, [])
                x.append(pid)
                qrels[qid] = x
        self.qrels = qrels

        self.sample_num = config.sample_num - 1
        self.epoch = 0
        self.num_samples = len(self.top1000)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, qid, pids, sample_num):
        '''
        qid:int
        pids:list
        sample_num:int
        '''
        pids = [pid for pid in pids if pid not in self.qrels[qid]]
        pids = pids[self.min_index:self.max_index]
        interval = len(pids) // sample_num
        offset = self.epoch % interval
        sample_pids = pids[offset::interval][:sample_num]
        return sample_pids

    def __getitem__(self, idx):
        cols = self.top1000[idx]
        qid = cols[0]
        pids = list(cols[1]['pid'])
        sample_neg_pids = self.sample(qid, pids, self.sample_num)
        pos_id = random.choice(self.qrels.get(qid))
        query = self.query.loc[qid]['text']
        data = [(query, self.collection.loc[pos_id]['para'])]
        for neg_pid in sample_neg_pids:
            data.append((query, self.collection.loc[neg_pid]['para']))
        return data

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        for qp_pairs in sample_list:
            for q, p in qp_pairs:
                qrys.append(q)
                psgs.append(p)
        features = self.tokenizer(qrys,
                                  psgs,
                                  padding=True,
                                  truncation=True,
                                  return_tensors="pt",
                                  max_length=self.max_seq_len)
        return features

    def __len__(self):
        return self.num_samples


class CrossEncoderDevDataset(Dataset):

    def __init__(self, config: CrossEncoderConfig):
        self.max_seq_len = config.max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path)

        self.collection = pd.read_csv(t2ranking_data / config.collection,
                                      sep="\t",
                                      quoting=3)
        self.collection.columns = ['pid', 'para']
        self.collection = self.collection.fillna("NA")
        self.collection.index = self.collection.pid
        self.collection.pop('pid')

        self.query = pd.read_csv(t2ranking_data / config.dev_query, sep="\t")
        self.query.columns = ['qid', 'text']
        self.query.index = self.query.qid
        self.query.pop('qid')
        self.top1000 = pd.read_csv(t2ranking_data / config.dev_top1000,
                                   sep="\t",
                                   header=None)
        self.num_samples = len(self.top1000)

    def __getitem__(self, idx):
        cols = self.top1000.iloc[idx]
        qid = cols[0]
        pid = cols[1]
        return self.query.loc[qid]['text'], self.collection.loc[pid][
            'para'], qid, pid

    def _collate_fn(self, sample_list):
        qrys = []
        psgs = []
        qids = []
        pids = []
        for q, p, qid, pid in sample_list:
            qrys.append(q)
            psgs.append(p)
            qids.append(qid)
            pids.append(pid)
        features = self.tokenizer(qrys,
                                  psgs,
                                  padding=True,
                                  truncation=True,
                                  return_tensors="pt",
                                  max_length=self.max_seq_len)
        return features, {"qids": np.array(qids), "pids": np.array(pids)}

    def __len__(self):
        return self.num_samples
