output_dim=2
hidden_dim=100
filter_size=40
seq_len=120
num_feature=2
batchsize=60
epochs=100

step=15

num_critic=10


#data="data/NYISO_Jul_spread_load.csv"
#dataset="NYISO_spread"

data="data/CAISO_both.csv"
dataset="CAISO_both"

# the “best” values you picked
lrG=0.00001
gp_coef_inn=0.1
gp_coef_recons=0.1
coef_recons=0.1
seed=1

# run it once
python main.py \
  -data_path $data \
  -dataset $dataset \
  -output_dim $output_dim \
  -hidden_dim $hidden_dim \
  -seq_len $seq_len \
  -num_feature $num_feature \
  -filter_size $filter_size \
  -batch_size $batchsize \
  -epochs $epochs \
  -pred_step $step \
  -num_critic $num_critic \
  -lrD $lrG \
  -lrG $lrG \
  -gp_coef_inn $gp_coef_inn \
  -gp_coef_recons $gp_coef_recons \
  -coef_recons $coef_recons \
  -seed $seed