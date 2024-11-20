#!/usr/bin/env bash

csv_file_address="/media/volume/Data/TCGA_Data/Clinical/gdc_sample_sheet.tsv"
img_dir="/media/volume/Data/TCGA_Data/Patches"
pt_dir="/media/volume/Data/TCGA_pt"
magnifications="5 10 20"
save_dir="/media/volume/Code/Trained-Models/Quilt-CALG/Third-Attn"

# backbone="hug_quilt"
# backbone_dim=768

# backbone="conch"
# backbone_dim=512

backbone="resnet50"
backbone_dim=2048

model_name="FirstAttn"
data_workers=12
prefetch_factor=3
grad_accum_steps=32
preload=0
num_folds=5
test_ratio=0.2
train_ratio=0.8
valid_ratio=0
batch_size=1
random_seed=7
num_epochs=100
continue_training=0
es_patience=20

learning_rate=3e-4
weight_decay=4e-6
lr_decay=0.5
loss_fn="nll_surv"
alpha_surv=0.2

embed_dim=256
num_heads=4
num_attn_layers=1

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --csv-file-address $csv_file_address \
    --img-dir $img_dir \
    --save-dir $save_dir \
    --model-name $model_name \
    --data-workers $data_workers \
    --prefetch-factor $prefetch_factor \
    --grad-accum-steps $grad_accum_steps \
    --preload $preload \
    --num-folds $num_folds \
    --test-ratio $test_ratio \
    --train-ratio $train_ratio \
    --valid-ratio $valid_ratio \
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
    --backbone-dim $backbone_dim