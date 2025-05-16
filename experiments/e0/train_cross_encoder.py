import functools
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import distributed
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import experiments.core.dataset as dataset
from experiments.core.config import CrossEncoderConfig, t2ranking_data
from experiments.core.msmarco_eval import calc_mrr
from experiments.e0.modeling import CrossEncoder

SEED = 2023
best_mrr = -1
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
print = functools.partial(print, flush=True)


def merge(eval_cnts, file_pattern='output/res.step-%d.part-0%d'):
    f_list = []
    total_part = torch.distributed.get_world_size()
    for part in range(total_part):
        f0 = open(file_pattern % (eval_cnts, part))
        f_list += f0.readlines()
    f_list = [l.strip().split("\t") for l in f_list]
    dedup = defaultdict(dict)
    for qid, pid, score in f_list:
        dedup[qid][pid] = float(score)
    mp = defaultdict(list)
    for qid in dedup:
        for pid in dedup[qid]:
            mp[qid].append((pid, dedup[qid][pid]))
    for qid in mp:
        mp[qid].sort(key=lambda x: x[1], reverse=True)
    with open(file_pattern.replace('.part-0%d', '') % eval_cnts, 'w') as f:
        for qid in mp:
            for idx, (pid, score) in enumerate(mp[qid]):
                f.write(
                    str(qid) + "\t" + str(pid) + '\t' + str(idx + 1) + "\t" +
                    str(score) + '\n')
    for part in range(total_part):
        os.remove(file_pattern % (eval_cnts, part))


def train_cross_encoder(config: CrossEncoderConfig, model, optimizer):
    epoch = 0
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        print(
            f'Starting training, up to {config.epochs} epochs, LR={config.learning_rate}'
        )

    train_dataset = dataset.CrossEncoderTrainDataset(config)
    dev_dataset = dataset.CrossEncoderDevDataset(config)

    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=config.dev_batch_size,
                            collate_fn=dev_dataset._collate_fn,
                            sampler=dev_sampler,
                            num_workers=4)

    # validate(model, dev_loader, epoch, config)

    for epoch in range(1, config.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=train_dataset._collate_fn,
                                  sampler=train_sampler,
                                  num_workers=4,
                                  drop_last=True)
        train_iteration(model, optimizer, train_loader, epoch, config)
        torch.distributed.barrier()
        #del train_loader
        #if epoch % 1 == 0:
        #    validate(model, dev_loader, epoch, config)
        #    torch.distributed.barrier()


def validate(model, dev_loader, epoch, config: CrossEncoderConfig):
    global best_mrr
    local_rank = torch.distributed.get_rank()
    with torch.no_grad():
        model.eval()
        scores_lst = []
        qids_lst = []
        pids_lst = []
        for record1, record2 in tqdm(dev_loader):
            with autocast('cuda',
                          dtype=config.torch_dtype,
                          enabled=config.dtype != "float32"):
                scores = model(_prepare_inputs(record1))
            qids = record2['qids']
            pids = record2['pids']
            scores_lst.append(scores.detach().cpu().numpy().copy())
            qids_lst.append(qids.copy())
            pids_lst.append(pids.copy())
        qids_lst = np.concatenate(qids_lst).reshape(-1)
        pids_lst = np.concatenate(pids_lst).reshape(-1)
        scores_lst = np.concatenate(scores_lst).reshape(-1)
        with open("output/res.step-%d.part-0%d" % (epoch, local_rank),
                  'w') as f:
            for qid, pid, score in zip(qids_lst, pids_lst, scores_lst):
                f.write(str(qid) + '\t' + str(pid) + '\t' + str(score) + '\n')
        torch.distributed.barrier()
        if local_rank == 0:
            merge(epoch)
            metrics = calc_mrr(t2ranking_data / config.dev_qrels,
                               'output/res.step-%d' % epoch)
            mrr = metrics['MRR @10']
            if mrr > best_mrr:
                print("*" * 50)
                print("new top")
                print("*" * 50)
                best_mrr = mrr
                torch.save(model.module.lm.state_dict(),
                           os.path.join(config.model_out_dir, "reranker.p"))


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()  #进程数
    return rt


def _prepare_inputs(record):
    prepared = {}
    local_rank = torch.distributed.get_rank()
    for key in record:
        x = record[key]
        if isinstance(x, torch.Tensor):
            prepared[key] = x.to(local_rank)
        else:
            prepared[key] = _prepare_inputs(x)
    return prepared


def train_iteration(model, optimizer, data_loader, epoch,
                    config: CrossEncoderConfig):
    model.train()
    total_loss = 0.
    local_rank = torch.distributed.get_rank()
    start = time.time()
    local_start = time.time()
    all_steps_per_epoch = len(data_loader)
    step = 0
    scaler = GradScaler()
    for record in tqdm(data_loader):
        record = _prepare_inputs(record)
        with autocast('cuda',
                      dtype=config.torch_dtype,
                      enabled=config.dtype != "float32"):
            loss = model(record)
        torch.distributed.barrier()
        reduced_loss = reduce_tensor(loss.data)
        total_loss += reduced_loss.item()
        # optimize
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step += 1
        if step % config.report == 0 and local_rank == 0:
            seconds = time.time() - local_start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            local_start = time.time()
            print(
                "epoch:%d training step: %d/%d, mean loss: %.5f, current loss: %.5f,"
                % (epoch, step, all_steps_per_epoch, total_loss / step,
                   loss.cpu().detach().numpy()),
                "report used time:%02d:%02d:%02d," % (h, m, s),
                end=' ')
            seconds = time.time() - start
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
            print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))
    if local_rank == 0:
        # model.save(os.path.join(args.model_out_dir, "weights.epoch-%d.p"%(epoch)))
        torch.save(
            model.module.state_dict(),
            os.path.join(config.model_out_dir, "weights.epoch-%d.p" % (epoch)))
        seconds = time.time() - start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print(f'train epoch={epoch} loss={total_loss}')
        print("total used time:%02d:%02d:%02d" % (h, m, s), end=' ')
        print(time.strftime("[TIME %Y-%m-%d %H:%M:%S]", time.localtime()))


def main(device):
    config = CrossEncoderConfig()

    model = CrossEncoder(config)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    params = {'params': [v for k, v in params]}
    optimizer = torch.optim.Adam([params],
                                 lr=config.learning_rate,
                                 weight_decay=0.0)

    model = DistributedDataParallel(model,
                                    device_ids=[local_rank],
                                    output_device=local_rank,
                                    find_unused_parameters=False)
    os.makedirs(config.model_out_dir, exist_ok=True)

    train_cross_encoder(config, model, optimizer)


if __name__ == '__main__':
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    main(device)
    '''
    batch_size = 8
    dtype = "float16"
    gradient_checkpoint = True
    99/25046 [02:39<11:03:18,  1.60s/it]
    epoch:1 training step: 100/25046, mean loss: 2.97111, current loss: 2.68750, report used time:00:02:41, total used time:00:02:41 [TIME 2025-05-16 18:10:54]   
    
    17.860Gi/23.988Gi
    '''
