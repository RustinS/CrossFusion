import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler, WeightedRandomSampler
from torchvision.transforms.functional import convert_image_dtype

EPSILON = 1.0e-10
UNKNOWN_VALUE = "Unknown"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def survival_labeler(row, years=5):
    if row["Overall Survival Status"] == "0:LIVING" or row["Overall Survival (Months)"] / 12 > years:
        return 0
    else:
        return 1


def get_data_list(csv_file, img_dir):
    info_df = pd.read_csv(csv_file, sep="\t")
    clinical_df = pd.read_csv("/media/volume/Data/TCGA_Data/Clinical/brca_tcga_pub_clinical_data.tsv", sep="\t")
    extra_info_df = pd.read_csv("/media/volume/Data/TCGA_Data/Clinical/TCGA_corrected.csv")[["Patient ID", "npn", "size"]]
    patch_gcn_df = pd.read_csv("/media/volume/Data/TCGA_Data/Clinical/tcga_brca_patch_gcn.csv")[["case_id", "censorship", "survival_months", "train"]]

    clinical_df = clinical_df.merge(info_df, left_on="Patient ID", right_on="Case ID", how="inner").merge(
        extra_info_df, on="Patient ID", how="left"
    )
    clinical_df = clinical_df.merge(patch_gcn_df, left_on="Patient ID", right_on="case_id", how="inner")

    clinical_df["image_file_name"] = clinical_df.apply(lambda row: f"{row['File Name'].split('.svs')[0]}", axis=1)

    clinical_df["5-years surv label"] = clinical_df.apply(survival_labeler, axis=1)

    image_names = []
    patch_file_names = []
    image_to_patch_dict = defaultdict(list)
    with Path(f"{img_dir}/info_file.txt").open("r") as info_file:
        for line in info_file:
            patch_file_name = line.strip()

            patch_file_name = f"{img_dir}/{patch_file_name.split('/')[-1]}"

            image_name = patch_file_name.split("/")[-1].split(".png")[0].split("_")[0]

            image_to_patch_dict[image_name].append(patch_file_name)
            patch_file_names.append(patch_file_name)
            image_names.append(image_name)

    keys_to_delete = []
    for image_name, patch_file_list in image_to_patch_dict.items():
        if len(patch_file_list) < 20:
            keys_to_delete.append(image_name)

    for key in keys_to_delete:
        del image_to_patch_dict[key]

    name_mapping_df = pd.DataFrame({"patch_name": patch_file_names, "image_name": image_names})

    clinical_df["Diagnosis Age"] = clinical_df["Diagnosis Age"].fillna(UNKNOWN_VALUE)
    clinical_df["Diagnosis Age"] = clinical_df["Diagnosis Age"].apply(lambda x: int(x) if isinstance(x, float) else x)

    clinical_df["ER Status"] = clinical_df["ER Status"].fillna(UNKNOWN_VALUE)
    clinical_df["ER Status"] = clinical_df["ER Status"].apply(
        lambda x: x if x in ["Positive", "Negative"] else UNKNOWN_VALUE
    )

    clinical_df["PR Status"] = clinical_df["PR Status"].fillna(UNKNOWN_VALUE)
    clinical_df["PR Status"] = clinical_df["PR Status"].apply(
        lambda x: x if x in ["Positive", "Negative"] else UNKNOWN_VALUE
    )

    clinical_df["HER2 Status"] = clinical_df["HER2 Status"].fillna(UNKNOWN_VALUE)
    clinical_df["HER2 Status"] = clinical_df["HER2 Status"].apply(
        lambda x: x if x in ["Positive", "Negative"] else UNKNOWN_VALUE
    )

    clinical_df["npn"] = clinical_df["npn"].fillna(UNKNOWN_VALUE)
    clinical_df["npn"] = clinical_df["npn"].apply(lambda x: int(x) if isinstance(x, float) else x)

    clinical_df["size"] = clinical_df["size"].fillna(UNKNOWN_VALUE)
    clinical_df["size"] = clinical_df["size"].apply(lambda x: int(x) if isinstance(x, float) else x)

    return (
        clinical_df,
        clinical_df["image_file_name"].to_list(),
        clinical_df["5-years surv label"].to_list(),
        name_mapping_df,
        image_to_patch_dict,
    )


class StratifiedSampler(Sampler):
    def __init__(self: Sampler, labels: list, batch_size: int = 10) -> None:
        self.batch_size = batch_size
        self.num_splits = int(len(labels) / self.batch_size)
        self.labels = labels
        self.num_steps = self.num_splits

    def _sampling(self: Sampler) -> list:
        skf = StratifiedKFold(n_splits=self.num_splits, shuffle=True)
        indices = np.arange(len(self.labels))

        return [tidx for _, tidx in skf.split(indices, self.labels)]

    def __iter__(self: Sampler) -> Iterator:
        return iter(self._sampling())

    def __len__(self: Sampler) -> int:
        return self.num_steps


def stratified_split(
    x: list,
    y: list,
    train: float,
    valid: float,
    test: float,
    num_folds: int,
    seed: int = 5,
) -> list:
    assert train + valid + test - 1.0 < EPSILON, "Ratios must sum to 1.0 ."

    outer_splitter = StratifiedShuffleSplit(
        n_splits=num_folds,
        train_size=train + valid,
        random_state=seed,
    )

    x = np.array(x)
    y = np.array(y)
    splits = []
    for train_idx, test_idx in outer_splitter.split(x, y):
        test_x = x[test_idx]
        test_y = y[test_idx]

        train_x = x[train_idx]
        train_y = y[train_idx]

        # Integrity check
        assert len(set(train_x).intersection(set(test_x))) == 0

        splits.append(
            {
                "train": list(zip(train_x, train_y)),
                "test": list(zip(test_x, test_y)),
            },
        )
    return splits

def postprocess(image_tensor):
    return convert_image_dtype(image_tensor, torch.uint8).squeeze().detach().cpu().permute(1, 2, 0).numpy()


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
                **kwargs,
            )
        else:
            loader = DataLoader(
                split_dataset,
                batch_size=batch_size,
                sampler=RandomSampler(split_dataset),
                worker_init_fn=worker_init_fn,
                generator=torch.Generator().manual_seed(opts.random_seed),
                **kwargs,
            )
    else:
        loader = DataLoader(
            split_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(split_dataset),
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(opts.random_seed),
            **kwargs,
        )

    return loader

def worker_init_fn(worker_id):
    """
    Initialize each DataLoader worker with a deterministic seed.

    Args:
        worker_id (int): Unique identifier for each worker.
    """
    # Retrieve the base seed from PyTorch
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
