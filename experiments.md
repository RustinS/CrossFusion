
#####################################################

backbone="hug_quilt"
model_name="SecondAttn"
grad_accum_steps=32
batch_size=1
random_seed=7
num_epochs=50

learning_rate=1e-5
weight_decay=2e-4
lr_decay=0.5
loss_fn="nll_surv"
alpha_surv=0.0

embed_dim=512
num_heads=8
num_attn_layers=6

|2024-08-30|18:29:46| - [LOGS] - Best Val C-Index List: [0.70883165 0.63268937 0.63092551 0.72921109 0.53961606] - Best Val Epoch List: [25, 13, 40, 15, 5]
|2024-08-30|18:29:46| - [LOGS] - TCGA-BRCA Complete C-Index: 0.648 +/- 0.067

#####################################################

backbone="hug_quilt"
model_name="SecondAttn"
grad_accum_steps=16
batch_size=1
random_seed=7
num_epochs=50

learning_rate=1e-5
weight_decay=2e-4
lr_decay=0.5
loss_fn="nll_surv"
alpha_surv=0.0

embed_dim=256
num_heads=4
num_attn_layers=6

|2024-09-20|03:51:20| - [LOGS] - Best Val C-Index List: [0.69963201 0.61820031 0.71557562 0.72153518 0.55532286] - Best Val Epoch List: [13, 12, 39, 22, 20]
|2024-09-20|03:51:20| - [LOGS] - TCGA-BRCA Complete C-Index: 0.662 +/- 0.065

##################################################### Sep 24

backbone="hug_quilt"
backbone_dim=768
model_name="FirstAttn"
grad_accum_steps=32
batch_size=1
random_seed=7
num_epochs=50

learning_rate=2e-5
weight_decay=2e-4
lr_decay=0.5
loss_fn="nll_surv"
alpha_surv=0.0

embed_dim=256
num_heads=4
num_attn_layers=1

|2024-09-24|12:56:32| - [LOGS] - Best Val C-Index List: [0.72677093 0.61667514 0.61286682 0.72110874 0.65933682] - Best Val Epoch List: [14, 1, 29, 26, 15]
|2024-09-24|12:56:32| - [LOGS] - TCGA-BRCA Complete C-Index: 0.667 +/- 0.049

##################################################### Oct 2

|2024-10-02|08:21:26| - [LOGS] - Best Val C-Index List: [0.71619135 0.68784952 0.55925508 0.68997868 0.64677138] - Best Val Epoch List: [30, 24, 41, 16, 34]
|2024-10-05|19:06:28| - [LOGS] - TCGA-BRCA Complete C-Index: 0.660 +/- 0.055

##################################################### Oct 5

magnifications="5 10 20"

backbone="hug_quilt"
backbone_dim=768

model_name="FirstAttn"
grad_accum_steps=16
random_seed=7
num_epochs=100
es_patience=30

learning_rate=1e-4
weight_decay=4e-6
lr_decay=0.5
loss_fn="nll_surv"
alpha_surv=0.2

embed_dim=256
num_heads=4
num_attn_layers=1

|2024-10-05|17:13:43| - [LOGS] - Best Val C-Index List: [0.713 0.663 0.619 0.744 0.631] - Best Val Epoch List: [33, 21, 46, 16, 22]
|2024-10-05|17:13:43| - [LOGS] - TCGA-BRCA Complete C-Index: 0.674 +/- 0.048

######################################################

backbone="conch"
backbone_dim=512

model_name="FirstAttn"
grad_accum_steps=16
random_seed=7
num_epochs=100
es_patience=30

learning_rate=1e-4
weight_decay=4e-6
lr_decay=0.5
loss_fn="nll_surv"
alpha_surv=0.2

embed_dim=256
num_heads=4
num_attn_layers=1

|2024-10-06|04:00:10| - [LOGS] - Best Val C-Index List: [0.683 0.751 0.634 0.646 0.631] - Best Val Epoch List: [18, 9, 9, 58, 58]
|2024-10-06|04:00:10| - [LOGS] - TCGA-BRCA Complete C-Index: 0.669 +/- 0.045

#####################################################

backbone="resnet50"
backbone_dim=2048

model_name="FirstAttn"
grad_accum_steps=32
random_seed=7
num_epochs=100
es_patience=30

learning_rate=3e-4
weight_decay=4e-6
lr_decay=0.5
loss_fn="nll_surv"
alpha_surv=0.2

embed_dim=256
num_heads=4
num_attn_layers=1

|2024-10-08|07:59:59| - [LOGS] - Best Val C-Index List: [0.666 0.664 0.621 0.69  0.602] - Best Val Epoch List: [16, 37, 7, 42, 7]
|2024-10-08|07:59:59| - [LOGS] - TCGA-BRCA Complete C-Index: 0.649 +/- 0.032