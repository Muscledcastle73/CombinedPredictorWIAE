#!/usr/bin/env bash
# --------- user-editable section ---------
lrG=0.0005          # learning-rate for both G and D
gp_coef_inn=0.3     # gradient-penalty (innovation)
gp_coef_recons=0.4  # gradient-penalty (reconstruction)
seed=42             # RNG seed
coef_recons=1.0     # λ in the paper (edit if you need it)


# --------- rarely changed ---------
output_dim=1
hidden_dim=100
filter_size=40
seq_len=120
num_feature=1
batchsize=60
epochs=100
pred_step=15
num_critic=10
data="data/CAISO_ACTUAL_FREQUENCY.csv"
dataset="CAISO_FREQUENCY"


python main.py \
 -data_path "$data" -dataset "$dataset" \
 -output_dim "$output_dim" -hidden_dim "$hidden_dim" -seq_len "$seq_len" \
 -num_feature "$num_feature" -filter_size "$filter_size" \
 -lrD "$lrG" -lrG "$lrG" \
 -batch_size "$batchsize" -epochs "$epochs" -num_critic "$num_critic" \
 -gp_coef_inn "$gp_coef_inn" -gp_coef_recons "$gp_coef_recons" \
 -coef_recons "$coef_recons" -seed "$seed" -pred_step "$pred_step"



