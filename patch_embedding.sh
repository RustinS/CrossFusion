#!/usr/bin/env bash

# BRCA
# wsi_path="/media/volume/Data/TCGA/BRCA-Diagnostic-Slides"
# patches_path="/media/volume/Data/TCGA/Patches"
# csv_path="/media/volume/Data/TCGA/gdc_sample_sheet.tsv"
# clinical_path="/media/volume/Data/TCGA_Data/Clinical/tcga_brca_patch_gcn.csv"

# LUAD
# wsi_path="/media/volume/Data/TCGA-LUAD/Slides"
# clinical_path="./data/tcga_luad_all_clean.csv.zip"
# csv_path="./data/TCGA-Manifests/gdc_manifest_LUAD.csv"
# splits_path="./data/splits/tcga_luad"
# patches_path="/media/volume/Data/TCGA-LUAD/Patches"
# output_dir="/media/volume/Data/TCGA-LUAD/Features"

# BLCA
# wsi_path="/projects/patho5nobackup/TCGA/TCGA_Data/BLCA"
# clinical_path="./data/tcga_blca_all_clean.csv.zip"
# csv_path="./data/TCGA-Manifests/gdc_manifest_BLCA.csv"
# patches_path="/projects/patho5nobackup/TCGA/Survival_Data/BLCA/Patches"
# splits_path="./data/splits/tcga_blca"
# output_dir="/projects/patho5nobackup/TCGA/Survival_Data/BLCA/Features"

# UCEC
wsi_path="/projects/patho5nobackup/TCGA/TCGA_Data/UCEC"
clinical_path="./data/tcga_ucec_all_clean.csv.zip"
csv_path="./data/TCGA-Manifests/gdc_manifest_UCEC.csv"
patches_path="/projects/patho5nobackup/TCGA/Survival_Data/UCEC/Patches"
splits_path="./data/splits/tcga_ucec"
output_dir="/projects/patho5nobackup/TCGA/Survival_Data/UCEC/Features"

# backbone="resnet50"
# backbone="conch"
# backbone="uni"
# backbone="hug_quilt"
backbone="prov_gigapath"

patch_size=512
batch_size=256
# batch_size=128
num_workers=4
embed_dim=2048
seed=0

CUDA_VISIBLE_DEVICES=2 python patch_embedding.py \
    --wsi_path $wsi_path \
    --patches_path $patches_path \
    --csv_path $csv_path \
    --clinical_path $clinical_path \
    --output_dir $output_dir \
    --backbone $backbone \
    --patch_size $patch_size \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --embed_dim $embed_dim \
    --seed $seed \
    --splits_path $splits_path