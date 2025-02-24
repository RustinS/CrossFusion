import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from utils.print_utils import get_curr_time_stamp, text_colors


def create_pbar(subset_name: str, num_steps: int) -> tqdm:
    time_stamp = get_curr_time_stamp()
    log_str = (
        "["
        + text_colors["logs"]
        + text_colors["bold"]
        + "LOGS"
        + text_colors["end_color"]
        + "]"
    )
    pbar_format = f"{time_stamp} - {log_str} - Processing: |{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}]"

    if subset_name == "train":
        pbar_format += " loss={postfix[1][loss]:0.5f}"
        pbar = tqdm(
            total=num_steps,
            leave=True,
            initial=0,
            bar_format=pbar_format,
            ascii=True,
            postfix=["", {"loss": float("NaN")}],
        )
    else:
        pbar = tqdm(total=num_steps, leave=True, bar_format=pbar_format, ascii=True)
    return pbar


def set_random_seed(seed, device):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def get_training_args():
    parser = argparse.ArgumentParser(description="Fine-tuning for survival prediction")

    parser.add_argument("--csv-path", type=str, help="Address of CSV manifest file.")
    parser.add_argument("--clinical-path", type=str, help="Address of clinical data.")
    parser.add_argument("--img-dir", type=str, help="Directory of images.")
    parser.add_argument("--save-dir", type=str, help="Directory to save results.")
    parser.add_argument("--model-name", type=str, help="Model name.")
    parser.add_argument("--data-workers", type=int, help="Number of data workers.")
    parser.add_argument("--prefetch-factor", type=int, help="Prefetch factor.")
    parser.add_argument("--grad-accum-steps", type=int, help="Gradient accumulation steps.")
    parser.add_argument("--preload", type=int, help="Preload flag.")
    parser.add_argument("--num-folds", type=int, help="Number of folds for cross-validation.")
    parser.add_argument("--splits-path", type=str, help="Directory for splits.")
    parser.add_argument("--batch-size", type=int, help="Batch size for training.")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs for training.")
    parser.add_argument("--continue-training", type=int, help="Continue training.")
    parser.add_argument("--learning-rate", type=float, help="Learning rate.")
    parser.add_argument("--lr-decay", type=float, help="Learning rate decay.")
    parser.add_argument("--weight-decay", type=float, help="Weight decay.")
    parser.add_argument("--loss-fn", type=str, help="Loss function.")
    parser.add_argument("--embed-dim", type=int, help="Embedding dimension.")
    parser.add_argument("--num-heads", type=int, help="Number of attention heads.")
    parser.add_argument("--num-attn-layers", type=int, help="Number of attention layers.")
    parser.add_argument("--pt-dir", type=str, help="Directory to the base pt folder.")
    parser.add_argument("--backbone", type=str, help="Feature extraction backbone.")
    parser.add_argument("--magnifications", type=int, nargs="+", help="Levels for patch extraction [5]")
    parser.add_argument("--backbone-dim", type=int, help="Backbone features dimension.")
    parser.add_argument("--alpha-surv", type=float, help="Alpha for survival loss.")
    parser.add_argument("--es-patience", type=int, help="Early stopping patience.")
    parser.add_argument("--dataset-name", type=str, help="Dataset name.")
    parser.add_argument("--warmup-epochs", type=int, help="Number of warmup epochs.")
    parser.add_argument("--bfloat16", type=int, help="Wether to use bfloat16.")

    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)

    args.magnifications = tuple(args.magnifications)

    if args.data_workers > 0:
        torch.set_num_threads(args.data_workers)

    return args

def get_eval_args():
    parser = argparse.ArgumentParser(description="Fine-tuning for survival prediction")

    parser.add_argument("--csv-path", type=str, help="Address of CSV manifest file.")
    parser.add_argument("--clinical-path", type=str, help="Address of clinical data.")
    parser.add_argument("--img-dir", type=str, help="Directory of images.")
    parser.add_argument("--save-dir", type=str, help="Directory to save results.")
    parser.add_argument("--model-name", type=str, help="Model name.")
    parser.add_argument("--data-workers", type=int, help="Number of data workers.")
    parser.add_argument("--prefetch-factor", type=int, help="Prefetch factor.")
    parser.add_argument("--num-folds", type=int, help="Number of folds for cross-validation.")
    parser.add_argument("--splits-path", type=str, help="Directory for splits.")
    parser.add_argument("--batch-size", type=int, help="Batch size for training.")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("--loss-fn", type=str, help="Loss function.")
    parser.add_argument("--embed-dim", type=int, help="Embedding dimension.")
    parser.add_argument("--num-heads", type=int, help="Number of attention heads.")
    parser.add_argument("--num-attn-layers", type=int, help="Number of attention layers.")
    parser.add_argument("--pt-dir", type=str, help="Directory to the base pt folder.")
    parser.add_argument("--backbone", type=str, help="Feature extraction backbone.")
    parser.add_argument("--magnifications", type=int, nargs="+", help="Levels for patch extraction [5]")
    parser.add_argument("--backbone-dim", type=int, help="Backbone features dimension.")
    parser.add_argument("--alpha-surv", type=float, help="Alpha for survival loss.")
    parser.add_argument("--dataset-name", type=str, help="Dataset name.")

    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)

    args.magnifications = tuple(args.magnifications)

    if args.data_workers > 0:
        torch.set_num_threads(args.data_workers)

    return args
