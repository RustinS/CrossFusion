#!/usr/bin/env bash

dataset_name=BRCA
clinical_path="./data/tcga_brca_all_clean2.csv.zip"
csv_path="./data/TCGA-Manifests/gdc_manifest_BRCA.csv"
splits_path="./data/splits/tcga_brca"
img_dir="/projects/patho5nobackup/TCGA/Survival_Data/BRCA/patches_clam"
pt_dir="/projects/patho5nobackup/TCGA/Survival_Data/BRCA/features_clam"
save_dir="/projects/patho5nobackup/TCGA/Trained-Models/CrossFusion/BRCA"


backbone="resnet50"
backbone_dim=2048

# backbone="conch"
# backbone_dim=512

# backbone="uni"
# backbone_dim=1536

# backbone="hug_quilt"
# backbone_dim=768

# backbone="prov_gigapath"
# backbone_dim=1536

model_name="CrossFusion"

magnifications="5 10 20"
data_workers=8
prefetch_factor=3
num_folds=5
batch_size=1
random_seed=7

loss_fn="nll_surv"
alpha_surv=0.2

embed_dim=512
num_heads=4
num_attn_layers=1

CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python heatmap.py \
    --csv-path $csv_path \
    --clinical-path $clinical_path \
    --img-dir $img_dir \
    --save-dir $save_dir \
    --model-name $model_name \
    --data-workers $data_workers \
    --prefetch-factor $prefetch_factor \
    --num-folds $num_folds \
    --splits-path $splits_path \
    --batch-size $batch_size \
    --random-seed $random_seed \
    --loss-fn $loss_fn \
    --embed-dim $embed_dim \
    --num-heads $num_heads \
    --num-attn-layers $num_attn_layers \
    --pt-dir $pt_dir \
    --backbone $backbone \
    --magnifications $magnifications \
    --alpha-surv $alpha_surv \
    --backbone-dim $backbone_dim \
    --dataset-name $dataset_name \