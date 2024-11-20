
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from utils.print_utils import print_error_message, print_info_message  # type: ignore


class TCGADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_names,
        csv_file,
        x20_pt_folder,
        x10_pt_folder,
        x5_pt_folder,
        image_to_patch_dict,
        preprocess,
        tokenizer,
        name,
    ):
        super(TCGADataset, self).__init__()

        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.x20_pt_folder = x20_pt_folder
        self.x10_pt_folder = x10_pt_folder
        self.x5_pt_folder = x5_pt_folder

        img_names = [img_name for img_name in img_names if img_name in image_to_patch_dict]

        self.clinical_df = csv_file[csv_file["image_file_name"].isin(img_names)]

        self.image_to_patch_dict = image_to_patch_dict
        self.image_names = img_names

        value_counts = self.clinical_df["5-years surv label"].value_counts().to_dict()

        info_str = f"Samples in {name} dataset: {len(self.image_names)} WSIs "
        info_str += f" (0: {value_counts[0]} | 1: {value_counts[1]})"
        print_info_message(info_str)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]

        x20_pt_file_path = f"{self.x20_pt_folder}/{image_name}.pt"
        x10_pt_file_path = f"{self.x10_pt_folder}/{image_name}.pt"
        x5_pt_file_path = f"{self.x5_pt_folder}/{image_name}_features.pt"


        if Path(x20_pt_file_path).exists():
            x20_patches_tensor = torch.load(x20_pt_file_path, map_location='cpu')
        else:
            print_error_message(f"File {x20_pt_file_path} does not exist.")

        if Path(x10_pt_file_path).exists():
            x10_patches_tensor = torch.load(x10_pt_file_path, map_location='cpu')
        else:
            print_error_message(f"File {x10_pt_file_path} does not exist.")

        if Path(x5_pt_file_path).exists():
            x5_patches_tensor = torch.load(x5_pt_file_path, map_location='cpu')
        else:
            print_error_message(f"File {x5_pt_file_path} does not exist.")

        text_data = self.clinical_df.loc[self.clinical_df["image_file_name"] == image_name].iloc[0]

        text = f"H&E image of a {text_data['Diagnosis Age']} years old patient with "
        text += f"{text_data['ER Status']} ER, "
        text += f"{text_data['PR Status']} PR, "
        text += f"{text_data['HER2 Status']} HER2, "
        text += f"{text_data['npn']} NPN, "
        text += f"and {text_data['size']} Tumor Size."

        # Tokenize the text
        inputs = self.tokenizer(text)
        input_ids = inputs.squeeze(0)

        label = text_data["5-years surv label"]

        return {
            "wsi_name": image_name,
            "x20_patches": x20_patches_tensor,
            "x10_patches": x10_patches_tensor,
            "x5_patches": x5_patches_tensor,
            "input_ids": input_ids,
            "label": label,
            "time_to_event": text_data["Overall Survival (Months)"] / 12
            if not pd.isna(text_data["Overall Survival (Months)"])
            else 0.0,
        }


# Index(['Study ID', 'Patient ID', 'Sample ID_x', 'Diagnosis Age', 'Cancer Type',
#        'Cancer Type Detailed', 'CN Cluster', 'Converted Stage', 'ER Status',
#        'Fraction Genome Altered', 'HER2 Status',
#        'Integrated Clusters (no exp)', 'Integrated Clusters (unsup exp)',
#        'Integrated Clusters (with PAM50)', 'Metastasis', 'Metastasis-Coded',
#        'Methylation Cluster', 'MIRNA Cluster', 'Mutation Count', 'Node',
#        'Node-Coded', 'Oncotree Code', 'Overall Survival (Months)',
#        'Overall Survival Status', 'PAM50 subtype', 'PR Status', 'RPPA Cluster',
#        'Number of Samples Per Patient', 'Sample Type_x', 'Sex',
#        'SigClust Intrinsic mRNA', 'SigClust Unsupervised mRNA',
#        'Somatic Status', 'Survival Data Form', 'TMB (nonsynonymous)',
#        'Tumor Stage', 'Tumor--T1 Coded', 'File ID', 'File Name',
#        'Data Category', 'Data Type', 'Project ID', 'Case ID', 'Sample ID_y',
#        'Sample Type_y', 'npn', 'size', 'image_file_name',
#        '5-years surv label'],
#       dtype='object')
