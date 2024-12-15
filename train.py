import json
import os
import traceback
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

from datasets.dataset_survival import GenericMILSurvivalDataset
from models.AMIL import AMIL
from models.CoCoFusion import CoCoFusion
from models.CoCoFusionConcat import CoCoFusionConcat
from models.CoCoFusionX import CoCoFusionX
from models.CoCoFusionXX import CoCoFusionXX
from models.DSMIL import DSMIL
from models.SCMIL import SCMIL
from models.TransMIL import TransMIL
from utils.data_utils import get_split_loader
from utils.general_utils import create_pbar, get_training_args, set_random_seed
from utils.print_utils import print_info_message, print_log_message
from utils.train_utils import (
    CoxSurvLoss,
    CrossEntropySurvLoss,
    NLLSurvLoss,
    print_network,
)
from utils.valid_utils import print_val_info_str


def build_model(opts):
    if opts.model_name == "CoCoFusion":
        model = CoCoFusion(
            embed_dim=opts.embed_dim,
            num_heads=opts.num_heads,
            num_layers=opts.num_attn_layers,
            backbone_dim=opts.backbone_dim,
            n_classes=opts.n_classes,
        )
    elif opts.model_name == "CoCoFusionConcat":
        model = CoCoFusionConcat(
            embed_dim=opts.embed_dim,
            num_heads=opts.num_heads,
            num_layers=opts.num_attn_layers,
            backbone_dim=opts.backbone_dim,
            n_classes=opts.n_classes,
        )
    elif opts.model_name == "CoCoFusionX":
        model = CoCoFusionX(
            embed_dim=opts.embed_dim,
            num_heads=opts.num_heads,
            num_layers=opts.num_attn_layers,
            backbone_dim=opts.backbone_dim,
            n_classes=opts.n_classes,
        )
    elif opts.model_name == "CoCoFusionXX":
        model = CoCoFusionXX(
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
    elif opts.model_name == "SCMIL":
        model = SCMIL(input_size=opts.backbone_dim, n_classes=opts.n_classes, hidden_size=opts.embed_dim)

    model = model.to("cuda:0").to(torch.bfloat16)
    model = torch.nn.DataParallel(model)

    # print_network(model)

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

    train_split, val_split = datasets
    print_info_message("Training on {} samples".format(len(train_split)))
    print_info_message("Validating on {} samples".format(len(val_split)))

    print_log_message("Init loss function...", empty_line=True)
    loss_fn = build_loss_fn(opts)
    scaler = torch.amp.GradScaler("cuda")

    print_log_message("Init Model...")
    opts.n_classes = 4
    model = build_model(opts)

    print_log_message("Init optimizer ...")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=opts.learning_rate, weight_decay=opts.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.num_epochs // 5, gamma=0.5)

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
            train_single_epoch(model, train_loader, optimizer, loss_fn, scheduler, opts.grad_accum_steps, scaler)

            val_c_index = validate_single_epoch(model, val_loader, loss_fn)

            if val_c_index > best_val_cindex and epoch > opts.warmup_epochs - 1:
                best_val_epoch = epoch
                best_val_cindex = val_c_index
                print_log_message("New Best Val C-Index ...")
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


def train_single_epoch(
    model, loader, optimizer, loss_fn, scheduler, grad_accum_steps, scaler
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.0

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    pbar = create_pbar("train", len(loader))

    for batch_idx, batch in enumerate(loader):
        x20_patches = batch["x20_patches"].to(device).to(torch.bfloat16)
        x10_patches = batch["x10_patches"].to(device).to(torch.bfloat16)
        x5_patches = batch["x5_patches"].to(device).to(torch.bfloat16)
        label = batch["label"].long().to(device)
        event_time = batch["event_time"].float()
        censorship = batch["censorship"].to(device)

        # with torch.amp.autocast("cuda", dtype=torch.float32):
        hazards, S, Y_hat, logits, _ = model(x5_patches, x10_patches, x20_patches)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=censorship)
        loss_value = loss.item()
        loss = loss / grad_accum_steps

        risk = -torch.sum(S.float(), dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk.item()
        all_censorships[batch_idx] = censorship.cpu().numpy().item()
        all_event_times[batch_idx] = event_time

        train_loss += loss_value

        pbar.postfix[1]["loss"] = train_loss / (batch_idx + 1)

        pbar.update()

        # scaler.scale(loss).backward()
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            optimizer.zero_grad()

    pbar.close()
    scheduler.step()

    train_loss /= len(loader)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08
    )[0]

    print_log_message(
        "Train Loss: {:.3f} - Train C-Index: {:.3f}".format(
            train_loss, c_index
        )
    )


def validate_single_epoch(
    model,
    loader,
    loss_fn=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.0
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    pbar = create_pbar("val", len(loader))
    for batch_idx, batch in enumerate(loader):
        x20_patches = batch["x20_patches"].to(device).to(torch.bfloat16)
        x10_patches = batch["x10_patches"].to(device).to(torch.bfloat16)
        x5_patches = batch["x5_patches"].to(device).to(torch.bfloat16)
        label = batch["label"].long().to(device)
        event_time = batch["event_time"]
        censorship = batch["censorship"].to(device)

        # with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float32):
        with torch.no_grad():
            hazards, S, Y_hat, _, _ = model(x5_patches, x10_patches, x20_patches)

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
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08
    )[0]

    print_log_message(
        "Val Loss: {:.3f} - Val C-Index: {:.3f}".format(
            val_loss, c_index
        )
    )

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
        best_val_cindex_list.append(round(best_val_cindex, 3))
        best_val_epoch_list.append(best_val_epoch)
        print_log_message(f"Best Val C-Index List: {best_val_cindex_list} - Current Best Val Epoch List: {best_val_epoch_list}")

    best_val_cindex_list = np.array(best_val_cindex_list)
    mean_c_index = np.mean(best_val_cindex_list)
    std_c_index = np.std(best_val_cindex_list)
    print_log_message(f"TCGA-{args.dataset_name} with {args.backbone} Backbone and {args.model_name} Model Complete C-Index: {mean_c_index:.3f} +/- {std_c_index:.3f}", empty_line=True)
