import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from conch.open_clip_custom import create_model_from_pretrained
from efficientnet_pytorch import EfficientNet
from einops import rearrange
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPVisionModel

from utils.print_utils import print_error_message, print_log_message

Image.MAX_IMAGE_PIXELS = None
VIEWER_SLIDE_NAME = "slide"


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_wsi_transform(patch_size, backbone=None):
    if backbone == "resnet50":
        transformlist = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif backbone == "vit_b_16":
        transformlist = transforms.Compose(
            [
                transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif backbone == "conch":
        _, transformlist = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token="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"
        )
    elif "hug" not in backbone.split("_"):
        transformlist = transforms.Compose(
            [
                transforms.RandomResizedCrop(patch_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif backbone == "hug_dinov2":
        transformlist = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    elif backbone == "hug_quilt":
        transformlist = AutoProcessor.from_pretrained("wisdomik/QuiltNet-B-32")
    else:
        transformlist = None
    return transformlist


def setup(args):
    model = None
    if args.backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in model.parameters():
            param.requires_grad = False
        args.embed_dim = model.fc.in_features
        model.fc = torch.nn.Identity()

    if args.backbone == "conch":
        model, _ = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token="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"
        )
        args.embed_dim = 512

    if args.backbone == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        for param in model.parameters():
            param.requires_grad = False
        args.embed_dim = model.heads.head.in_features
        model.heads.head = torch.nn.Identity()

    if args.backbone == "hug_dinov2":
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        for param in model.parameters():
            param.requires_grad = False
        args.embed_dim = 768

    if args.backbone == "hug_quilt":
        model = CLIPVisionModel.from_pretrained("wisdomik/QuiltNet-B-32")
        for param in model.parameters():
            param.requires_grad = False
        args.embed_dim = 768

    return args, model


def get_model(args, device="cuda"):
    args, model = setup(args)
    model = model.to(device)
    return model


class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self, patch_dir, wsi_name, mag_level, preprocess, backbone="hug_quilt"):
        super(EmbedDataset, self).__init__()

        self.mag_level = mag_level
        self.preprocess = preprocess
        self.base_dir = os.path.join(patch_dir, f"{mag_level}x")
        self.wsi_name = wsi_name
        self.backbone = backbone

        self.patch_path_list = glob.glob(os.path.join(self.base_dir, wsi_name, "*.jpeg"))

        print_log_message(f"Mag Level: {self.mag_level} - Number of Patches: {len(self.patch_path_list)}")

    def __len__(self):
        return len(self.patch_path_list)

    def __getitem__(self, index):
        patch_path = self.patch_path_list[index]

        patch = Image.open(patch_path)

        if "hug" in self.backbone.split("_"):
            patch = self.preprocess(images=patch, return_tensors="pt").pixel_values
        else:
            patch = self.preprocess(patch)
        return patch


def get_loader(args, wsi_name, mag_level):
    transform = get_wsi_transform(args.patch_size, backbone=args.backbone)
    dataset = EmbedDataset(args.patches_path, wsi_name, mag_level, transform, args.backbone)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return data_loader


def embed(args, model, wsi_name, wsi_patch_loader, backbone, mag_output_base, device="cuda"):
    model.eval()
    patch_emb_list = []

    for batch in tqdm(wsi_patch_loader):
        patch_batch = batch.to(device)
        with torch.no_grad():
            if backbone == "efficientnet-b1":
                bs = patch_batch.shape[0]
                patch_emb = model.extract_features(patch_batch)
                patch_emb = torch.nn.functional.adaptive_avg_pool2d(patch_emb, output_size=(1))
                patch_emb = patch_emb.view(bs, -1)
            elif "hug" in args.backbone.split("_"):
                patch_batch = patch_batch.squeeze(dim=1)
                patch_emb = model(pixel_values=patch_batch)
                patch_emb = patch_emb.pooler_output
            elif backbone == "conch":
                with torch.inference_mode():
                    patch_emb = model.encode_image(patch_batch, proj_contrast=False, normalize=False)
            else:
                patch_emb = model(patch_batch)

            patch_emb_list.append(patch_emb.cpu())
    wsi_emb = torch.cat(patch_emb_list, dim=0)
    assert wsi_emb.shape[1] == args.embed_dim, f"Not of correct Embed size , is {wsi_emb.shape}, " f"but should be {args.embed_dim}"
    torch.save(wsi_emb, os.path.join(mag_output_base, f"{wsi_name}.pt"))


def get_wsi_names(args, desired_levels):
    info_df = pd.read_csv(args.csv_path, sep="\t")
    clinical_df = pd.read_csv(args.clinical_path)[["case_id", "censorship", "survival_months", "slide_id"]]
    clinical_df = clinical_df.merge(info_df, left_on="slide_id", right_on="filename", how="inner")

    split_data = pd.read_csv(os.path.join(args.splits_path, "splits_0.csv"))
    all_case_ids = (
        split_data["train"].dropna().reset_index(drop=True).tolist() + split_data["val"].dropna().reset_index(drop=True).tolist()
    )

    mask = clinical_df["case_id"].isin(all_case_ids)
    clinical_df = clinical_df[mask].reset_index(drop=True)

    wsi_path_list = clinical_df.apply(lambda row: f"{args.wsi_path}/{row['id']}/{row['filename']}", axis=1).to_numpy()
    wsi_name_list = clinical_df.apply(lambda row: f"{row['filename'].split('.svs')[0]}", axis=1).to_numpy()

    missing_data = []
    print_log_message("Checking for missing WSI data:")
    for i in tqdm(range(len(wsi_path_list))):
        if not Path(wsi_path_list[i]).is_file():
            missing_data.append(i)

    if len(missing_data) > 0:
        print_error_message(f"Number of missing data: {len(missing_data)}")

    wsi_path_list = np.delete(wsi_path_list, missing_data)
    wsi_name_list = np.delete(wsi_name_list, missing_data)

    missing_data = []
    for mag_level in desired_levels:
        mag_dir = os.path.join(args.patches_path, f"{mag_level}x")
        print_log_message(f"Checking for missing patches in level {mag_level}x:")
        for i, wsi_name in tqdm(enumerate(wsi_name_list)):
            if not os.path.exists(os.path.join(mag_dir, wsi_name)) and i not in missing_data:
                print_log_message(f"Missing patches for {wsi_name} at {mag_level}x magnification level")
                if i not in missing_data:
                    missing_data.append(i)

    if len(missing_data) > 0:
        print_error_message(f"Number of missing data: {len(missing_data)}")

    wsi_name_list = np.delete(wsi_name_list, missing_data)

    return wsi_name_list


def main(args):
    set_seeds(args)
    desired_levels = tuple(args.magnifications)

    output_base = os.path.join(args.output_dir, args.backbone)

    if not os.path.exists(output_base):
        os.makedirs(output_base)

    for level in desired_levels:
        mag_output_base = os.path.join(output_base, f"{level}x")
        if not os.path.exists(mag_output_base):
            os.makedirs(mag_output_base)

    wsi_name_list = get_wsi_names(args, desired_levels)

    for idx, wsi_name in enumerate(wsi_name_list):
        print_log_message(f"Processing {idx + 1}/{len(wsi_name_list)} ({wsi_name}):", empty_line=True)
        for level in desired_levels:
            mag_output_base = os.path.join(output_base, f"{level}x")
            if not os.path.exists(os.path.join(mag_output_base, f"{wsi_name}.pt")):
                loader = get_loader(args, wsi_name, level)
                model = get_model(args)
                embed(args, model, wsi_name, loader, args.backbone, mag_output_base)
            else:
                print_log_message(f"Already processed at {level}x magnification level. Skipping...")


def set_seeds(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


def get_opts(parser):
    group = parser.add_argument_group("General details")

    group.add_argument("--wsi_path", type=str, default="/media/volume/Data/TCGA/BRCA-Diagnostic-Slides", help="Path to WSIs")
    group.add_argument("--patches_path", type=str, default="/media/volume/Data/TCGA/Patches", help="Path to Patches main dir")
    group.add_argument("-m", "--magnifications", type=int, nargs="+", default=(5, 10, 20), help="Levels for patch extraction [5]")
    group.add_argument("--csv_path", type=str, default="/media/volume/Data/TCGA/gdc_sample_sheet.tsv", help="Path to info CSV")
    group.add_argument("--splits_path", type=str, default="./data/splits/tcga_brca", help="Path to splits CSV files")
    group.add_argument(
        "--clinical_path", type=str, default="/media/volume/Data/TCGA_Data/Clinical/tcga_brca_patch_gcn.csv", help="Path to clinical CSV"
    )
    group.add_argument(
        "--backbone",
        default="vit_b_16",
        choices=["resnet50", "hug_quilt", "hug_dinov2", "conch", "vit_b_16"],
        help="pretrained network to use",
        type=str,
    )
    group.add_argument("-p", "--patch_size", type=int, choices=[224, 256, 512, 1024, 3136, 4096, 16384], default=512)
    group.add_argument("-b", "--batch_size", default=256, type=int, help="Batch size, if 1 will have indv rc names")
    group.add_argument("--num_workers", default=4, type=int)
    group.add_argument("--output_dir", type=str, default="/media/volume/Data/TCGA_pt", help="Path to output folder")
    group.add_argument("-e", "--embed_dim", default=2048, type=int, help="Embed dimension to spit out")
    group.add_argument("--seed", default=0, type=int)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Patch Embedding")
    parser = get_opts(parser)
    args = parser.parse_args()

    main(args)
    print_log_message("finished!")
