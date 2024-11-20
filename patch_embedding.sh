#!/usr/bin/env bash

# wsi_path="/media/volume/Data/TCGA/BRCA-Diagnostic-Slides"
# patches_path="/media/volume/Data/TCGA/Patches"
# csv_path="/media/volume/Data/TCGA/gdc_sample_sheet.tsv"
# clinical_path="/media/volume/Data/TCGA_Data/Clinical/tcga_brca_patch_gcn.csv"

wsi_path="/media/volume/Data/TCGA-LUAD/Slides"
patches_path="/media/volume/Data/TCGA-LUAD/Patches"
csv_path="./data/TCGA-Manifests/gdc_manifest_LUAD.csv"
clinical_path="./data/tcga_luad_all_clean.csv.zip"
splits_path="./data/splits/4foldcv/tcga_luad"

backbone="resnet50"

patch_size=512
batch_size=256
num_workers=4
output_dir="/media/volume/Data/TCGA-LUAD/Features"
embed_dim=2048
seed=0

python patch_embedding.py \
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