import os
import traceback
from argparse import Namespace

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

from datasets.dataset_survival import GenericMILSurvivalDataset
from models.AMIL import AMIL
from models.CrossFusion import CrossFusion
from models.CrossFusionConcat import CrossFusionConcat
from models.CrossFusionSingle import CrossFusionSingle
from models.DSMIL import DSMIL
from models.TransMIL import TransMIL
from utils.data_utils import get_split_loader
from utils.general_utils import create_pbar, get_training_args, set_random_seed
from utils.print_utils import print_info_message, print_log_message
from utils.train_utils import (
    CoxSurvLoss,
    CrossEntropySurvLoss,
    NLLSurvLoss,
)


def build_model(opts):
    if opts.model_name == "CrossFusion":
        model = CrossFusion(
            embed_dim=opts.embed_dim,
            num_heads=opts.num_heads,
            num_layers=opts.num_attn_layers,
            backbone_dim=opts.backbone_dim,
            n_classes=opts.n_classes,
        )
    elif opts.model_name == "CrossFusionConcat":
        model = CrossFusionConcat(
            embed_dim=opts.embed_dim,
            num_heads=opts.num_heads,
            num_layers=opts.num_attn_layers,
            backbone_dim=opts.backbone_dim,
            n_classes=opts.n_classes,
        )
    elif opts.model_name == "CrossFusionSingle":
        model = CrossFusionSingle(
            embed_dim=opts.embed_dim,
            num_heads=opts.num_heads,
            num_layers=opts.num_attn_layers,
            backbone_dim=opts.backbone_dim,
            n_classes=opts.n_classes,
        )
    elif opts.model_name == "AMIL":
        model = AMIL(
            backbone_dim=opts.backbone_dim,
            n_classes=opts.n_classes,
            gate=True,
        )
    elif opts.model_name == "TransMIL":
        model = TransMIL(
            backbone_dim=opts.backbone_dim,
            n_classes=opts.n_classes,
        )
    elif opts.model_name == "DSMIL":
        model = DSMIL(
            backbone_dim=opts.backbone_dim,
            n_classes=opts.n_classes,
        )

    model = model.cuda()
    model = torch.nn.DataParallel(model)

    return model


def build_loss_fn(opts):
    if opts.loss_fn == "ce_surv":
        loss_fn = CrossEntropySurvLoss(alpha=opts.alpha_surv)
    elif opts.loss_fn == "nll_surv":
        loss_fn = NLLSurvLoss(alpha=opts.alpha_surv)
    elif opts.loss_fn == "cox_surv":
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError
    return loss_fn


def train(datasets: tuple, fold_idx: int, opts: Namespace):
    print_log_message(f"Training Fold {fold_idx}", empty_line=True)

    split_save_dir = os.path.join(opts.save_dir, opts.model_name, opts.backbone, f"fold_{fold_idx}")
    if not os.path.exists(split_save_dir):
        os.makedirs(split_save_dir)

    train_split, val_split = datasets
    print_info_message("Training on {} samples".format(len(train_split)))
    print_info_message("Validating on {} samples".format(len(val_split)))

    print_log_message("Init loss function...", empty_line=True)
    loss_fn = build_loss_fn(opts)

    print_log_message("Init Model...")
    opts.n_classes = 4
    model = build_model(opts)
    dtype = torch.float16 if opts.bfloat16 else torch.float32

    print_log_message("Init optimizer ...")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=opts.learning_rate, weight_decay=opts.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=opts.learning_rate // 10, last_epoch=-1)

    print_log_message("Init Loaders...")
    train_loader = get_split_loader(
        train_split,
        opts,
        training=True,
        weighted=True,
        batch_size=opts.batch_size,
    )
    val_loader = get_split_loader(val_split, opts, batch_size=opts.batch_size)

    es_patience = opts.es_patience
    best_val_cindex = 0.0
    best_val_epoch = 0
    try:
        for epoch in range(opts.num_epochs):
            print_log_message(f"Epoch {epoch}:", empty_line=True)
            train_single_epoch(model, train_loader, optimizer, loss_fn, scheduler, opts.grad_accum_steps, dtype)

            val_c_index = validate_single_epoch(model, val_loader, loss_fn, dtype)

            if val_c_index > best_val_cindex and epoch + 1 > opts.warmup_epochs:
                best_val_epoch = epoch
                best_val_cindex = val_c_index
                print_log_message("New Best Val C-Index ...")
                model.module.save(os.path.join(split_save_dir, "best_model.pt"))
                print_log_message("Saved the model.")
            else:
                print_log_message(f"No importvement in Val C-Index ({epoch - best_val_epoch}/{es_patience}) ...")
                if epoch - best_val_epoch >= es_patience:
                    print_log_message("Early stopping ...")
                    break
            print_log_message(f"Fold Best Val C-Index: {best_val_cindex:.3f} - Best Val Epoch: {best_val_epoch}")
    except Exception as e:
        traceback.print_exc()
        print_log_message(f"Error: {e}")

    print_info_message("Best Val C-Index: {:.4f}".format(best_val_cindex))
    print("\n")
    return best_val_cindex, best_val_epoch


def train_single_epoch(model, loader, optimizer, loss_fn, scheduler, grad_accum_steps, dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.0

    print_log_message(f"Current LR: {scheduler.get_last_lr()[0]}")

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    pbar = create_pbar("train", len(loader))

    for batch_idx, batch in enumerate(loader):
        with torch.amp.autocast('cuda', dtype=dtype):
            x20 = batch["x20_patches"].to(device)
            x10 = batch["x10_patches"].to(device)
            x5 = batch["x5_patches"].to(device)

            label = batch["label"].long().to(device)
            event_time = batch["event_time"].float()
            censorship = batch["censorship"].to(device)

            hazards, S, _, _, _ = model(x5, x10, x20)
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=censorship)
            loss_value = loss.item()
            loss = loss / grad_accum_steps

            risk = -torch.sum(S.float(), dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk.item()
            all_censorships[batch_idx] = censorship.cpu().numpy().item()
            all_event_times[batch_idx] = event_time

            train_loss += loss_value

            pbar.postfix[1]["loss"] = train_loss / (batch_idx + 1)

            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        pbar.update()

    pbar.close()
    scheduler.step()

    train_loss /= len(loader)

    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print_log_message("Train Loss: {:.3f} - Train C-Index: {:.3f}".format(train_loss, c_index))


def validate_single_epoch(
    model,
    loader,
    loss_fn=None,
    dtype=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.0
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    pbar = create_pbar("val", len(loader))
    for batch_idx, batch in enumerate(loader):
        with torch.amp.autocast('cuda', dtype=dtype):
            x20 = batch["x20_patches"].to(device)
            x10 = batch["x10_patches"].to(device)
            x5 = batch["x5_patches"].to(device)

            label = batch["label"].long().to(device)
            event_time = batch["event_time"]
            censorship = batch["censorship"].to(device)

            with torch.no_grad():
                hazards, S, _, _, _ = model(x5, x10, x20)

                loss = loss_fn(hazards=hazards.float(), S=S.float(), Y=label, c=censorship, alpha=0)
                loss_value = loss.item()

            risk = -torch.sum(S.float(), dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk.item()
            all_censorships[batch_idx] = censorship.cpu().numpy().item()
            all_event_times[batch_idx] = event_time

            val_loss += loss_value

        pbar.update()
    pbar.close()

    val_loss /= len(loader)
    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print_log_message("Val Loss: {:.3f} - Val C-Index: {:.3f}".format(val_loss, c_index))

    return c_index


if __name__ == "__main__":
    args = get_training_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_info_message(f"Training {args.model_name} model...")

    dataset = GenericMILSurvivalDataset(
        args=args,
        clinical_path=args.clinical_path,
        shuffle=False,
        print_info=True,
        patient_strat=False,
        n_bins=4,
        label_col="survival_months",
    )

    best_val_cindex_list = []
    best_val_epoch_list = []
    for fold_idx in range(args.num_folds):
        set_random_seed(args.random_seed, device)
        print("\n")
        train_dataset, val_dataset = dataset.return_splits(os.path.join(args.splits_path, f"splits_{fold_idx}.csv"))

        datasets = (train_dataset, val_dataset)

        best_val_cindex, best_val_epoch = train(datasets, fold_idx, args)
        best_val_cindex_list.append(round(float(best_val_cindex), 3))
        best_val_epoch_list.append(best_val_epoch)
        print_log_message(f"Best Val C-Index List: {best_val_cindex_list} - Current Best Val Epoch List: {best_val_epoch_list}")

    best_val_cindex_list = np.array(best_val_cindex_list)
    mean_ci = np.mean(best_val_cindex_list)
    std_ci = np.std(best_val_cindex_list)
    print_log_message(
        f"{args.dataset_name} with {args.backbone} Backbone and {args.model_name} Model Complete C-Index: {mean_ci:.3f} +/- {std_ci:.3f}",
        empty_line=True,
    )
