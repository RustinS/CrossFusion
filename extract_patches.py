import argparse
import logging
import multiprocessing
import os
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils
from PIL import Image, ImageFilter, ImageStat
from skimage.morphology import opening
from tiatoolbox.tools import patchextraction
from tiatoolbox.wsicore.wsireader import WSIReader
from torchvision import transforms
from tqdm import tqdm

from utils.print_utils import print_error_message, print_log_message

logging.getLogger().setLevel(logging.CRITICAL)


class TCGADatasetPNG(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_file, clinical_file, splits_path):
        if not Path(csv_file).is_file():
            print_error_message("{} does not exist.".format(csv_file))

        super(TCGADatasetPNG, self).__init__()

        info_df = pd.read_csv(csv_file, sep="\t")
        clinical_df = pd.read_csv(clinical_file)[["case_id", "slide_id"]]
        clinical_df = clinical_df.merge(info_df, left_on="slide_id", right_on="filename", how="inner")

        split_data = pd.read_csv(os.path.join(splits_path, "splits_0.csv"))
        all_case_ids = (
            split_data["train"].dropna().reset_index(drop=True).tolist() + split_data["val"].dropna().reset_index(drop=True).tolist()
        )

        mask = clinical_df["case_id"].isin(all_case_ids)
        clinical_df = clinical_df[mask].reset_index(drop=True)

        wsi_path_list = clinical_df.apply(lambda row: f"{img_dir}/{row['id']}/{row['filename']}", axis=1).to_numpy()
        wsi_name_list = clinical_df.apply(lambda row: f"{row['filename'].split('.svs')[0]}", axis=1).to_numpy()

        missing_data = []
        print_log_message("Checking for missing data:")
        for i in tqdm(range(len(wsi_path_list))):
            if not Path(wsi_path_list[i]).is_file():
                missing_data.append(i)

        if len(missing_data) > 0:
            print_error_message("Missing data: {}".format([wsi_name_list[i] for i in missing_data]))

        self.wsi_path_list = np.delete(wsi_path_list, missing_data)
        self.wsi_name_list = np.delete(wsi_name_list, missing_data)

        if not Path(self.wsi_path_list[0]).is_file():
            print_error_message("{} file does not exist.".format(self.wsi_path_list[0]))

        print_log_message("Samples in dataset: {}".format(len(self.wsi_path_list)))

    def __len__(self):
        return len(self.wsi_path_list)

    def __getitem__(self, index):
        return self.wsi_path_list[index], self.wsi_name_list[index]


class CALGBDatasetPNG(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_file):
        if not Path(csv_file).is_file():
            print_error_message("{} does not exist.".format(csv_file))

        super(CALGBDatasetPNG, self).__init__()

        clinical_df = pd.read_csv(csv_file).dropna(subset=["survstat", "survyrs"])

        wsi_path_list = clinical_df["Image File Name"].apply(lambda x: img_dir + "/" + x).to_numpy()

        wsi_name_list = clinical_df["Image File Name"].apply(lambda x: x.split(".")[0]).to_numpy()

        missing_data = []
        print_log_message("Checking for missing data:")
        for i in tqdm(range(len(wsi_path_list))):
            if not Path(wsi_path_list[i]).is_file():
                missing_data.append(i)

        self.wsi_path_list = np.delete(wsi_path_list, missing_data)
        self.wsi_name_list = np.delete(wsi_name_list, missing_data)

        if not Path(self.wsi_path_list[0]).is_file():
            print_error_message("{} file does not exist.".format(self.wsi_path_list[0]))

        print_log_message("Samples in dataset: {}".format(len(self.wsi_path_list)))

    def __len__(self):
        return len(self.wsi_path_list)

    def __getitem__(self, index):
        return self.wsi_path_list[index], self.wsi_name_list[index]


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool"):
    r = rgb[:, 0, :, :] < red_upper_thresh
    g = rgb[:, 1, :, :] > green_lower_thresh
    b = rgb[:, 2, :, :] > blue_lower_thresh
    result = ~(r & g & b)

    if output_type == "float":
        result = result.float()
    elif output_type == "uint8":
        result = result.byte() * 255

    return result


def filter_green_pen(rgb, output_type="bool"):
    return (
        filter_green(rgb, 150, 160, 140, output_type)
        & filter_green(rgb, 70, 110, 110, output_type)
        & filter_green(rgb, 45, 115, 100, output_type)
        & filter_green(rgb, 30, 75, 60, output_type)
        & filter_green(rgb, 195, 220, 210, output_type)
        & filter_green(rgb, 225, 230, 225, output_type)
        & filter_green(rgb, 170, 210, 200, output_type)
        & filter_green(rgb, 20, 30, 20, output_type)
        & filter_green(rgb, 50, 60, 40, output_type)
        & filter_green(rgb, 30, 50, 35, output_type)
        & filter_green(rgb, 65, 70, 60, output_type)
        & filter_green(rgb, 100, 110, 105, output_type)
        & filter_green(rgb, 165, 180, 180, output_type)
        & filter_green(rgb, 140, 140, 150, output_type)
        & filter_green(rgb, 185, 195, 195, output_type)
    )


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool"):
    r = rgb[:, 0, :, :] < red_upper_thresh
    g = rgb[:, 1, :, :] < green_upper_thresh
    b = rgb[:, 2, :, :] > blue_lower_thresh
    result = ~(r & g & b)

    if output_type == "float":
        result = result.float()
    elif output_type == "uint8":
        result = result.byte() * 255

    return result


def filter_blue_pen(rgb, output_type="bool"):
    return (
        filter_blue(rgb, 60, 120, 190, output_type)
        & filter_blue(rgb, 120, 170, 200, output_type)
        & filter_blue(rgb, 175, 210, 230, output_type)
        & filter_blue(rgb, 145, 180, 210, output_type)
        & filter_blue(rgb, 37, 95, 160, output_type)
        & filter_blue(rgb, 30, 65, 130, output_type)
        & filter_blue(rgb, 130, 155, 180, output_type)
        & filter_blue(rgb, 40, 35, 85, output_type)
        & filter_blue(rgb, 30, 20, 65, output_type)
        & filter_blue(rgb, 90, 90, 140, output_type)
        & filter_blue(rgb, 60, 60, 120, output_type)
        & filter_blue(rgb, 110, 110, 175, output_type)
    )


def mask_rgb(rgb, mask):
    mask = mask.unsqueeze(0).expand_as(rgb)  # Expands mask to match rgb's dimensions
    return rgb * mask


class WSIDataset(torch.utils.data.Dataset):
    def __init__(self, patch_coords, wsi_data, patch_size, mag_level, threshold=15, power=-1):
        super(WSIDataset, self).__init__()
        self.patch_coords = patch_coords
        self.wsi_path, self.wsi_name = wsi_data
        self.patch_size = patch_size
        self.threshold = threshold
        self.mag_level = mag_level
        self.power = power

        self.black_threshold = 5
        self.white_threshold = 250

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, index):
        patch_coord = self.patch_coords[index]

        wsi = WSIReader.open(input_img=self.wsi_path) if self.power == -1 else WSIReader.open(input_img=self.wsi_path, power=self.power)
        patch = wsi.read_bounds(patch_coord, resolution=self.mag_level, units="power", coord_space="resolution")
        wsi.openslide_wsi.close()

        patch = self.process_patch(patch, self.black_threshold, self.white_threshold, self.threshold, self.patch_size)

        if patch is not None:
            return self.transform(patch), patch_coord
        else:
            return None, None

    @staticmethod
    def process_patch(patch, black_threshold=5, white_threshold=240, threshold=15, patch_size=512):
        element224 = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        patch = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to("cuda")
        mask_no_green_pen = filter_green_pen(patch, output_type="bool")
        mask_no_blue_pen = filter_blue_pen(patch, output_type="bool")
        mask_pens = mask_no_green_pen & mask_no_blue_pen

        patch = mask_rgb(
            patch,
            torch.tensor(opening(mask_pens.squeeze(0).cpu().numpy(), element224.cpu().numpy()), device="cuda"),
        )
        patch = torch.where(patch < black_threshold, torch.tensor(255.0, device="cuda"), patch)
        patch = torch.where(patch > white_threshold, torch.tensor(255.0, device="cuda"), patch)

        patch = patch.squeeze().permute(1, 2, 0).cpu().numpy()
        patch = Image.fromarray(patch.astype("uint8"))

        edge = patch.filter(ImageFilter.FIND_EDGES)
        edge = ImageStat.Stat(edge).sum
        edge = np.mean(edge) / (patch_size**2)

        w, h = patch.size
        if edge > threshold:
            if not (w == patch_size and h == patch_size):
                patch = patch.resize((patch_size, patch_size))
            return patch
        return None


def collate_fn(batch):
    batch = [(img.cpu(), lst) for img, lst in batch if img is not None and lst is not None]

    if len(batch) == 0:
        return torch.tensor([]), []

    images, lists = zip(*batch)

    images_tensor = torch.stack(images, dim=0)

    return images_tensor, lists
    # batch = [item.cpu() for item in batch if item is not None]
    # if len(batch) == 0:
    #     return torch.tensor([])
    # return torch.stack(batch, dim=0)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_opts(parser):
    group = parser.add_argument_group("general details")

    group.add_argument("--wsi_path", type=str, default="/media/volume/Data/TCGA/BRCA-Diagnostic-Slides", help="Path to WSIs")
    group.add_argument("--csv_path", type=str, default="/media/volume/Data/TCGA/gdc_sample_sheet.tsv", help="Path to info CSV")
    group.add_argument("--dataset", type=str, default="TCGA", help="Dataset name [TCGA]")
    group.add_argument(
        "--clinical_path", type=str, default="/media/volume/Data/TCGA_Data/Clinical/tcga_brca_patch_gcn.csv", help="Path to clinical CSV"
    )
    group.add_argument("--splits_path", type=str, default="./data/splits/tcga_brca", help="Path to splits CSV files")
    group.add_argument("--output_dir", type=str, default="/media/volume/Data/TCGA/Patches", help="Output directory to save patches")

    group.add_argument("--num_workers", type=int, default=4, help="Number of workers for Data Loader [0]")

    group.add_argument("--patch_size", type=int, default=512, help="Patch Size [512]")
    group.add_argument("--thresh", type=int, default=15, help="Threshold to allow patches with background [15]")
    group.add_argument("--batch_size", type=int, default=16, help="Batch Size to process the patches [32]")
    group.add_argument("-m", "--magnifications", type=int, nargs="+", default=(5, 10, 20), help="Levels for patch extraction [5]")

    return parser


def create_mag_dirs(args, mag_output_dirs, desired_levels):
    for mag, mag_output_dir in mag_output_dirs.items():
        if mag not in desired_levels:
            continue
        mag_dir = os.path.join(args.output_dir, mag_output_dir)
        if not os.path.isdir(mag_dir):
            os.makedirs(mag_dir)


def is_already_processed(args, mag_output_dirs, mag_levels, desired_levels, wsi_name):
    already_processed = True
    for mag_level in mag_levels:
        if mag_level not in desired_levels:
            continue
        output_path = os.path.join(args.output_dir, mag_output_dirs[mag_level], wsi_name)

        if os.path.exists(output_path):
            continue
        else:
            already_processed = False
    return already_processed


def generate_extra_patches(args, min_patch_nums, wsi, mask, mag_level, patch_coords, patch_tensors_list):
    print_log_message(f"Number of patches is less than {min_patch_nums[mag_level]}. Regenerating ...")
    min_x = min([coord[0] for coord in patch_coords])
    min_y = min([coord[1] for coord in patch_coords])
    max_x = max([coord[0] for coord in patch_coords])
    max_y = max([coord[1] for coord in patch_coords])

    mask_thumb = mask.slide_thumbnail(resolution=1.25, units="power")

    while len(patch_tensors_list) < min_patch_nums[mag_level]:
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)

        patch_mask = mask_thumb[int(x / 4) : int(x / 4) + args.patch_size, int(y / 4) : int(y / 4) + args.patch_size]
        ones_ratio = np.mean(patch_mask)
        if ones_ratio < 0.01:
            continue

        patch = wsi.read_rect((x, y), (args.patch_size, args.patch_size), resolution=mag_level, units="power", coord_space="resolution")
        patch = WSIDataset.process_patch(patch, threshold=args.thresh, patch_size=args.patch_size)

        if patch is not None:
            patch_tensors_list.append(patch)
            new_coord = np.array([[x, y, x + args.patch_size, y + args.patch_size]])
            patch_coords = np.concatenate((patch_coords, new_coord), axis=0)
            if len(patch_tensors_list) == min_patch_nums[mag_level]:
                print_log_message(f"Found new patch. Number of patches: {len(patch_tensors_list)}")
            else:
                print_log_message(f"Found new patch. Number of patches: {len(patch_tensors_list)}", end="\r")

    print_log_message("Rearanging patches ...")
    patches_with_coords = list(zip(patch_tensors_list, patch_coords))

    sorted_patches_with_coords = sorted(patches_with_coords, key=lambda x: (x[1][0], x[1][1]))

    sorted_patches, sorted_coordinates = zip(*sorted_patches_with_coords)

    patch_tensors_list = list(sorted_patches)
    sorted_coordinates = list(sorted_coordinates)
    return patch_tensors_list, sorted_coordinates


def create_data_loader(WSIDataset, collate_fn, args, wsi_data, mag_level, patch_coords, num_workers=0, power=-1):
    wsi_dataset = WSIDataset(patch_coords, wsi_data, args.patch_size, mag_level, args.thresh, power)

    if num_workers > 0:
        return torch.utils.data.DataLoader(
            wsi_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            persistent_workers=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            prefetch_factor=1,
        )
    else:
        return torch.utils.data.DataLoader(
            wsi_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )


def save_patches(output_path, patch_tensors_list, patch_coords_list):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    to_pil = transforms.ToPILImage()
    for patch_idx, patch in enumerate(tqdm(patch_tensors_list)):
        if isinstance(patch, torch.Tensor):
            patch = to_pil(patch)
        patch.save(os.path.join(output_path, f"{patch_idx}.jpeg"))

    patch_coords = np.array(patch_coords_list)
    np.save(os.path.join(output_path, "coords.npy"), patch_coords)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Patch extraction for WSI")
    parser = get_opts(parser)
    args = parser.parse_args()

    mag_output_dirs = {5: "5x", 10: "10x", 20: "20x"}
    min_patch_nums = {5: 150, 10: 500, 20: 1000}
    err_patch_nums = {5: 5, 10: 15, 20: 50}
    x5_skip_patch_nums = 75 if args.dataset == "CALGB" else 0
    mag_levels = [5, 10, 20]
    desired_levels = tuple(args.magnifications)

    print_log_message(f"Desired magnifications: {desired_levels}")

    if args.dataset == "TCGA":
        dataset = TCGADatasetPNG(
            img_dir=args.wsi_path,
            csv_file=args.csv_path,
            clinical_file=args.clinical_path,
            splits_path=args.splits_path,
        )
    elif args.dataset == "CALGB":
        dataset = CALGBDatasetPNG(
            img_dir=args.wsi_path,
            csv_file=args.csv_path,
        )

    create_mag_dirs(args, mag_output_dirs, desired_levels)

    for idx, wsi_data in enumerate(dataset):
        print("\n")
        wsi_path, wsi_name = wsi_data
        should_process = True
        print_log_message(f"Processing {idx + 1}/{len(dataset)} ({wsi_name}):", empty_line=True)

        if is_already_processed(args, mag_output_dirs, mag_levels, desired_levels, wsi_name):
            print_log_message(f"Already Fully processed for the {desired_levels} magnification levels. Skipping ...")
            continue

        print_log_message("Generating tissue mask ...")
        try:
            wsi = WSIReader.open(input_img=wsi_path)
            mask = wsi.tissue_mask(method="morphological", resolution=1.25, units="power")
            power = -1
        except Exception as e:
            print_log_message(f"Error: {e}")
            print_log_message("Retrying with power 20.0 ...")
            wsi = WSIReader.open(input_img=wsi_path, power=20.0)
            mask = wsi.tissue_mask(method="morphological", resolution=1.25, units="power")
            power = 20.0

        for mag_level in mag_levels:
            if mag_level not in desired_levels or not should_process:
                continue
            print_log_message(f"Processing {mag_level}x magnification level ...", empty_line=True)
            output_path = os.path.join(args.output_dir, mag_output_dirs[mag_level], wsi_name)

            if os.path.exists(output_path):
                print_log_message("Already processed. Skipping ...")
                continue

            try:
                fixed_patch_extractor = patchextraction.get_patch_extractor(
                    input_img=wsi,
                    method_name="slidingwindow",
                    patch_size=(args.patch_size, args.patch_size),
                    stride=(args.patch_size, args.patch_size),
                    resolution=mag_level,
                    units="power",
                    input_mask=mask,
                    min_mask_ratio=0.01,
                )
            except Exception as e:
                print_log_message(f"Error: {e}")
                continue

            patch_coords = fixed_patch_extractor.coordinate_list

            print_log_message(f"Number of unfiltered patches: {len(patch_coords)}")

            if len(patch_coords) < x5_skip_patch_nums and mag_level == 5:
                print_log_message(f"Not enough patches ({len(patch_coords)}). Skipping ...")
                should_process = False
                continue

            if len(patch_coords) > 6000:
                data_loader = create_data_loader(
                    WSIDataset, collate_fn, args, wsi_data, mag_level, patch_coords, num_workers=0, power=power
                )
            else:
                data_loader = create_data_loader(
                    WSIDataset, collate_fn, args, wsi_data, mag_level, patch_coords, num_workers=args.num_workers, power=power
                )

            try:
                patch_tensors_list = []
                patch_coords_list = []
                for batch in tqdm(data_loader):
                    if batch[0].size(0) == 0:
                        continue
                    patch_tensors_list += batch[0].squeeze(dim=1)
                    patch_coords_list += batch[1]
            except Exception as e:
                traceback.print_exc()
                print_error_message(f"Error: {e}")
                del data_loader
                continue

            if len(patch_tensors_list) < err_patch_nums[mag_level]:
                print_log_message(f"Not enough patches ({len(patch_tensors_list)}). Skipping ...")
                should_process = False
                del data_loader
                continue

            print_log_message(f"Number of filtered patches: {len(patch_tensors_list)}")
            if len(patch_tensors_list) < min_patch_nums[mag_level]:
                patch_tensors_list, patch_coords_list = generate_extra_patches(args, min_patch_nums, wsi, mask, mag_level, patch_coords_list, patch_tensors_list)

            print_log_message("Saving patches ...")
            save_patches(output_path, patch_tensors_list, patch_coords_list)

        del wsi, mask
