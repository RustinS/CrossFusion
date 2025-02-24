import os

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
from utils.general_utils import create_pbar, get_eval_args, set_random_seed
from utils.print_utils import print_info_message, print_log_message
from utils.train_utils import (
    CoxSurvLoss,
    CrossEntropySurvLoss,
    NLLSurvLoss,
)
from utils.valid_utils import plot_kaplan_meier_curve


def build_model(opts, save_dir):
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

    state_dict = torch.load(os.path.join(save_dir, "best_model.pt"), weights_only=True)
    model.load_state_dict(state_dict)

    model = model.to("cuda")
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


def evaluate(val_split, fold_idx, opts):
    print_log_message(f"Evaluating Fold {fold_idx}", empty_line=True)

    split_save_dir = os.path.join(opts.save_dir, f"fold_{fold_idx}")

    print_info_message("{} Samples".format(len(val_split)))

    loss_fn = build_loss_fn(opts)

    opts.n_classes = 4
    model = build_model(opts, split_save_dir)

    val_loader = get_split_loader(val_split, opts, batch_size=opts.batch_size)

    c_index, risk_scores, censorships, event_times = validate_single_epoch(model, val_loader, loss_fn)

    print_info_message(f"Fold {fold_idx} Best Val C-Index: {c_index:.4f}")
    return c_index, risk_scores, censorships, event_times


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
        x20_patches = batch["x20_patches"].to(device)
        x10_patches = batch["x10_patches"].to(device)
        x5_patches = batch["x5_patches"].to(device)

        label = batch["label"].long().to(device)
        event_time = batch["event_time"]
        censorship = batch["censorship"].to(device)

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
    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    return c_index, all_risk_scores, all_censorships, all_event_times


if __name__ == "__main__":
    args = get_eval_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_info_message(f"Evaluating {args.model_name} model...")

    dataset = GenericMILSurvivalDataset(
        args=args,
        clinical_path=args.clinical_path,
        shuffle=False,
        print_info=True,
        patient_strat=False,
        n_bins=4,
        label_col="survival_months",
    )

    cindex_list = []
    risk_scores_list = []
    censorships_list = []
    event_times_list = []
    for fold_idx in range(args.num_folds):
        set_random_seed(args.random_seed, device)
        print("\n")
        _, val_dataset = dataset.return_splits(os.path.join(args.splits_path, f"splits_{fold_idx}.csv"))

        cindex, risk_scores, censorships, event_times = evaluate(val_dataset, fold_idx, args)

        cindex_list.append(round(float(cindex), 3))
        risk_scores_list.append(risk_scores)
        censorships_list.append(censorships)
        event_times_list.append(event_times)

    risk_scores_list = np.concatenate(risk_scores_list)
    censorships_list = np.concatenate(censorships_list)
    event_times_list = np.concatenate(event_times_list)

    plot_kaplan_meier_curve(
        risk_scores_list,
        censorships_list,
        event_times_list,
        args.dataset_name,
        os.path.join(args.save_dir, f"km_curve_{args.dataset_name}.png"),
    )

    cindex_list = np.array(cindex_list)
    mean_c_i = np.mean(cindex_list)
    std_c_i = np.std(cindex_list)
    print_log_message(
        f"TCGA-{args.dataset_name} with {args.backbone} Backbone and {args.model_name} Model C-Index: {mean_c_i:.3f} +/- {std_c_i:.3f}",
        empty_line=True,
    )
