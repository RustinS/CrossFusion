#!/usr/bin/env bash

wsi_path="/media/volume/Data/TCGA-LUAD/Slides"
clinical_path="./data/tcga_luad_all_clean.csv.zip"
csv_path="./data/TCGA-Manifests/gdc_manifest_LUAD.csv"
splits_path="./data/splits/4foldcv/tcga_luad"
dataset="TCGA"
output_dir="/media/volume/Data/TCGA-LUAD/Patches"

# wsi_path="/media/volume/Data/CALGB/raw"
# csv_path="/media/volume/Data/CALGB/clinical/complete_data_one_image.csv"
# dataset="CALGB"
# output_dir="/media/volume/Data/CALGB/Patches"

num_workers=4
batch_size=512
thresh=1
batch_size=16

python extract_patches.py \
    --wsi_path $wsi_path \
    --clinical_path $clinical_path \
    --csv_path $csv_path \
    --output_dir $output_dir \
    --num_workers $num_workers \
    --batch_size $batch_size \
    --thresh $thresh \
    --dataset $dataset \
    --splits_path $splits_path