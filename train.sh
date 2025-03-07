#!/usr/bin/env bash


# dataset_name=BLCA
# clinical_path="./data/tcga_blca_all_clean.csv.zip"
# csv_path="./data/TCGA-Manifests/gdc_manifest_BLCA.csv"
# splits_path="./data/splits/tcga_blca"
# img_dir="/projects/patho5nobackup/TCGA/Survival_Data/BLCA/patches_clam"
# pt_dir="/projects/patho5nobackup/TCGA/Survival_Data/BLCA/features_clam"
# save_dir="/projects/patho5nobackup/TCGA/Trained-Models/CrossFusion/BLCA"


# dataset_name=BRCA
# clinical_path="./data/tcga_brca_all_clean.csv.zip"
# csv_path="./data/TCGA-Manifests/gdc_manifest_BRCA.csv"
# splits_path="./data/splits/tcga_brca"
# img_dir="/projects/patho5nobackup/TCGA/Survival_Data/BRCA/patches_clam"
# pt_dir="/projects/patho5nobackup/TCGA/Survival_Data/BRCA/features_clam"
# save_dir="/projects/patho5nobackup/TCGA/Trained-Models/CrossFusion/BRCA"


# dataset_name=COAD
# clinical_path="./data/tcga_coad_all_clean.csv"
# csv_path="./data/TCGA-Manifests/gdc_manifest_COAD.csv"
# splits_path="./data/splits/tcga_coad"
# img_dir="/projects/patho5nobackup/TCGA/Survival_Data/COAD/patches_clam"
# pt_dir="/projects/patho5nobackup/TCGA/Survival_Data/COAD/features_clam"
# save_dir="/projects/patho5nobackup/TCGA/Trained-Models/CrossFusion/COAD"


# dataset_name=GBMLGG
# clinical_path="./data/tcga_gbmlgg_all_clean.csv.zip"
# csv_path="./data/TCGA-Manifests/gdc_manifest_GBMLGG.csv"
# splits_path="./data/splits/tcga_gbmlgg"
# img_dir="/projects/patho5nobackup/TCGA/Survival_Data/GBMLGG/patches_clam"
# pt_dir="/projects/patho5nobackup/TCGA/Survival_Data/GBMLGG/features_clam"
# save_dir="/projects/patho5nobackup/TCGA/Trained-Models/CrossFusion/GBMLGG"


# dataset_name=LUAD
# clinical_path="./data/tcga_luad_all_clean.csv.zip"
# csv_path="./data/TCGA-Manifests/gdc_manifest_LUAD.csv"
# splits_path="./data/splits/tcga_luad"
# img_dir="/projects/patho5nobackup/TCGA/Survival_Data/LUAD/patches_clam"
# pt_dir="/projects/patho5nobackup/TCGA/Survival_Data/LUAD/features_clam"
# save_dir="/projects/patho5nobackup/TCGA/Trained-Models/CrossFusion/LUAD"


dataset_name=UCEC
clinical_path="./data/tcga_ucec_all_clean.csv.zip"
csv_path="./data/TCGA-Manifests/gdc_manifest_UCEC.csv"
splits_path="./data/splits/tcga_ucec"
img_dir="/projects/patho5nobackup/TCGA/Survival_Data/UCEC/patches_clam"
pt_dir="/projects/patho5nobackup/TCGA/Survival_Data/UCEC/features_clam"
save_dir="/projects/patho5nobackup/TCGA/Trained-Models/CrossFusion/UCEC"


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

# model_name="CrossFusionConcat"
# model_name="CrossFusionSingle"

# model_name="AMIL"
# model_name="DSMIL"
# model_name="TransMIL"

magnifications="5 10 20"
data_workers=8
prefetch_factor=3
grad_accum_steps=32
preload=0
num_folds=5
batch_size=1
random_seed=7
num_epochs=100
warmup_epochs=5
continue_training=0
es_patience=20

learning_rate=1e-4

weight_decay=4e-6
lr_decay=0.5
loss_fn="nll_surv"
alpha_surv=0.2

embed_dim=512
num_heads=4
num_attn_layers=1
bfloat16=1

CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
    --csv-path $csv_path \
    --clinical-path $clinical_path \
    --img-dir $img_dir \
    --save-dir $save_dir \
    --model-name $model_name \
    --data-workers $data_workers \
    --prefetch-factor $prefetch_factor \
    --grad-accum-steps $grad_accum_steps \
    --preload $preload \
    --num-folds $num_folds \
    --splits-path $splits_path \
    --batch-size $batch_size \
    --random-seed $random_seed \
    --num-epochs $num_epochs \
    --continue-training $continue_training \
    --es-patience $es_patience \
    --learning-rate $learning_rate \
    --weight-decay $weight_decay \
    --loss-fn $loss_fn \
    --embed-dim $embed_dim \
    --num-heads $num_heads \
    --num-attn-layers $num_attn_layers \
    --pt-dir $pt_dir \
    --backbone $backbone \
    --magnifications $magnifications \
    --lr-decay $lr_decay \
    --alpha-surv $alpha_surv \
    --backbone-dim $backbone_dim \
    --dataset-name $dataset_name \
    --warmup-epochs $warmup_epochs \
    --bfloat16 $bfloat16