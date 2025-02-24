import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def get_split_loader(split_dataset, opts, training=False, weighted=False, batch_size=1):
    kwargs = {"num_workers": opts.data_workers} if device.type == "cuda" else {}
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            loader = DataLoader(
                split_dataset,
                batch_size=batch_size,
                sampler=WeightedRandomSampler(weights, len(weights)),
                worker_init_fn=worker_init_fn,
                generator=torch.Generator().manual_seed(opts.random_seed),
                pin_memory=True,
                persistent_workers=True,
                **kwargs,
            )
        else:
            loader = DataLoader(
                split_dataset,
                batch_size=batch_size,
                sampler=RandomSampler(split_dataset),
                worker_init_fn=worker_init_fn,
                generator=torch.Generator().manual_seed(opts.random_seed),
                pin_memory=True,
                persistent_workers=True,
                **kwargs,
            )
    else:
        loader = DataLoader(
            split_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(split_dataset),
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(opts.random_seed),
            pin_memory=True,
            persistent_workers=True,
            **kwargs,
        )

    return loader


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
