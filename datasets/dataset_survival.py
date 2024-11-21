from __future__ import division, print_function

import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.print_utils import print_error_message, print_info_message, print_log_message


class GenericWSISurvivalDataset(Dataset):
    def __init__(
        self,
        args,
        clinical_path="",
        shuffle=False,
        print_info=True,
        n_bins=4,
        patient_strat=False,
        label_col="survival_months",
        eps=1e-6,
    ):
        self.custom_test_ids = None
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.pt_dir = args.pt_dir

        self.pt_folders = {}
        for mag_level in args.magnifications:
            self.pt_folders[f"x{mag_level}"] = os.path.join(args.pt_dir, args.backbone, f"{mag_level}x")

        slide_data = self.prep_slide_data(clinical_path, args)

        assert label_col in slide_data.columns
        self.label_col = label_col

        if shuffle:
            np.random.seed(args.seed)
            np.random.shuffle(slide_data)

        patients_df = slide_data.copy()
        uncensored_df = patients_df[patients_df["censorship"] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        disc_labels, q_bins = pd.cut(
            patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True
        )
        patients_df.insert(2, "label", disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index("case_id")
        for patient in patients_df["case_id"]:
            slide_ids = slide_data.loc[patient, "slide_id"]
            slide_ids = np.array(slide_ids).reshape(-1) if isinstance(slide_ids, str) else slide_ids.values
            patient_dict.update({patient: slide_ids})

        self.patient_dict = patient_dict

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data["case_id"])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, "label"]
            slide_data.at[i, "disc_label"] = key
            censorship = slide_data.loc[i, "censorship"]
            key = (key, int(censorship))
            slide_data.at[i, "label"] = label_dict[key]

        self.bins = q_bins
        self.num_classes = len(self.label_dict)

        patients_df = slide_data.drop_duplicates(["case_id"])
        self.patient_data = {"case_id": patients_df["case_id"].values, "label": patients_df["label"].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:11]
        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def prep_slide_data(self, clinical_path, args):
        info_csv_path = args.csv_path

        info_df = pd.read_csv(info_csv_path, sep="\t")
        slide_data = pd.read_csv(clinical_path, index_col=0, low_memory=False)
        # slide_data = slide_data.merge(info_df, left_on="case_id", right_on="Case ID", how="inner")
        slide_data = slide_data.merge(info_df, left_on="slide_id", right_on="filename", how="inner")

        # slide_data["image_file_name"] = slide_data.apply(lambda row: f"{row['File Name'].split('.svs')[0]}", axis=1)
        slide_data["image_file_name"] = slide_data.apply(lambda row: f"{row['filename'].split('.svs')[0]}", axis=1).to_numpy()

        present_names = []
        for _, row in slide_data.iterrows():
            present = True
            for mag_level in args.magnifications:
                if not os.path.exists(os.path.join(self.pt_folders[f"x{mag_level}"], f"{row['image_file_name']}.pt")):
                    present = False
                    break
            if present:
                present_names.append(row["slide_id"])

        slide_data = slide_data[slide_data["slide_id"].isin(present_names)]

        if "IDC" in slide_data["oncotree_code"]:
            slide_data = slide_data[slide_data["oncotree_code"] == "IDC"]
        return slide_data

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data["label"] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data["label"] == i)[0]

    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data["case_id"]))  # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data["case_id"] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data["label"][locations[0]]  # get patient label
            patient_labels.append(label)

        self.patient_data = {"case_id": patients, "label": np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        return len(self.slide_data)

    def summarize(self):
        print_info_message(f"Number of slides: {len(self.slide_data)}")
        print_info_message(f"Label column: {self.label_col}")
        print_info_message(f"Number of classes: {self.num_classes}")
        print_info_message(f"Label dictionary: {self.label_dict}")
        print_info_message(f"Slide-level counts: {self.slide_data['label'].value_counts(sort=False).to_dict()}")

    def get_split_from_df(self, all_splits: dict, split_key: str = "train"):
        if split_key == "train":
            train_split = all_splits["train_slide_ids"]
            train_split = train_split.dropna().reset_index(drop=True).to_list()

            val_split = all_splits["val_slide_ids"]
            val_split = val_split.dropna().reset_index(drop=True).to_list()

            train_split += val_split
        else:
            split = all_splits[f"{split_key}_slide_ids"]
            split = split.dropna().reset_index(drop=True).to_list()

        if len(split) > 0:
            mask = self.slide_data["slide_id"].isin(split)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = GenericSplit(
                df_slice,
                metadata=self.metadata,
                label_col=self.label_col,
                patient_dict=self.patient_dict,
                num_classes=self.num_classes,
                pt_folders=self.pt_folders,
            )
        else:
            split = None

        return split

    def return_splits(self, csv_path):
        all_splits = pd.read_csv(csv_path)
        train_split = self.get_split_from_df(all_splits=all_splits, split_key="train")
        val_split = self.get_split_from_df(all_splits=all_splits, split_key="val")

        train_slide_ids = train_split.slide_data["slide_id"].to_list()
        val_slide_ids = val_split.slide_data["slide_id"].to_list()

        assert len(set(train_slide_ids) & set(val_slide_ids)) == 0, "Overlap between Train/Val splits"

        return train_split, val_split

    def get_list(self, ids):
        return self.slide_data["slide_id"][ids]

    def getlabel(self, ids):
        return self.slide_data["label"][ids]

    def __getitem__(self, idx):
        return None


class GenericMILSurvivalDataset(GenericWSISurvivalDataset):
    def __init__(self, **kwargs):
        super(GenericMILSurvivalDataset, self).__init__(**kwargs)

    def __getitem__(self, idx):
        case_id = self.slide_data["case_id"][idx]
        slide_id = self.slide_data["slide_id"][idx]
        label = self.slide_data["disc_label"][idx]
        event_time = self.slide_data[self.label_col][idx]
        censorship = self.slide_data["censorship"][idx]

        image_name = self.slide_data["image_file_name"][idx]

        x20_pt_file_path = os.path.join(self.pt_folders["x20"], f"{image_name}.pt")
        x10_pt_file_path = os.path.join(self.pt_folders["x10"], f"{image_name}.pt")
        x5_pt_file_path = os.path.join(self.pt_folders["x5"], f"{image_name}.pt")

        if os.path.exists(x20_pt_file_path):
            x20_patches_tensor = torch.load(x20_pt_file_path, map_location='cpu', weights_only=False)
        else:
            print_error_message(f"File {x20_pt_file_path} does not exist.")

        if os.path.exists(x10_pt_file_path):
            x10_patches_tensor = torch.load(x10_pt_file_path, map_location='cpu', weights_only=False)
        else:
            print_error_message(f"File {x10_pt_file_path} does not exist.")

        if os.path.exists(x5_pt_file_path):
            x5_patches_tensor = torch.load(x5_pt_file_path, map_location='cpu', weights_only=False)
        else:
            print_error_message(f"File {x5_pt_file_path} does not exist.")

        return {
            "case_id": case_id,
            "slide_id": slide_id,
            "x20_patches": x20_patches_tensor,
            "x10_patches": x10_patches_tensor,
            "x5_patches": x5_patches_tensor,
            "label": label,
            "event_time": event_time,
            "censorship": censorship,
        }


class GenericSplit(GenericMILSurvivalDataset):
    def __init__(self, slide_data, metadata, label_col=None, patient_dict=None, num_classes=2, pt_folders=None):
        self.slide_data = slide_data
        self.metadata = metadata
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.pt_folders = pt_folders
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data["label"] == i)[0]

    def __len__(self):
        return len(self.slide_data)
