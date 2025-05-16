import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))


def _prepare_inputs(record):
    prepared = {}
    for key in record:
        x = record[key]
        if isinstance(x, torch.Tensor):
            prepared[key] = x
        else:
            prepared[key] = _prepare_inputs(x)
    return prepared


def train_iteration(data_loader):
    for record in tqdm(data_loader):
        _prepare_inputs(record)


def validate(dev_loader):
    for record1, record2 in tqdm(dev_loader):
        _prepare_inputs(record1)


def main():
    import experiments.core.dataset as dataset
    from experiments.core.config import CrossEncoderConfig

    config = CrossEncoderConfig()

    train_dataset = dataset.CrossEncoderTrainDataset(config)
    dev_dataset = dataset.CrossEncoderDevDataset(config)

    print(len(train_dataset), len(dev_dataset))

    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    dev_loader = DataLoader(dev_dataset,
                            batch_size=config.dev_batch_size,
                            collate_fn=dev_dataset._collate_fn,
                            sampler=dev_sampler,
                            num_workers=1)

    for epoch in range(1, config.epochs + 1):
        print("epoch: ", epoch)
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=train_dataset._collate_fn,
                                  sampler=train_sampler,
                                  num_workers=1,
                                  drop_last=True)

        train_iteration(train_loader)

        if epoch % 1 == 0:
            validate(dev_loader)


if __name__ == '__main__':
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

    main()
    '''
    33/3130 [00:25<37:45,  1.37it/s]
    '''
