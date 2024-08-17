#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=24000
#SBATCH --job-name="gopt"
#SBATCH --output=../exp/log_%j.txt

set -x
stage=1
stop_stage=1000

lr=1e-3
batch_size=25
embed_dim=32
num_epochs=50
model=fluScorer
am=wav2vec2_large
cluster_pred=True

model(){
  fluScorerNoclu
  fluScorer
}

exp_dir=exp/flu-${lr}-${depth}-${batch_size}-${embed_dim}-${model}-${am}-nogelu-br

# repeat times
repeat_list=(0 1 2 3 4)
seed_list=(0 11 22 33 44)

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for repeat in "${repeat_list[@]}"; do
        mkdir -p $exp_dir/${repeat}
        python3 train.py \
            --lr ${lr} \
            --exp-dir ${exp_dir}/${repeat} \
            --batch_size ${batch_size} --embed_dim ${embed_dim} \
            --model ${model} --am ${am} --n-epochs ${num_epochs} \
            --cluster_pred ${cluster_pred} \
			--seed "${seed_list[$repeat]}" 
    done
    python collect_summary.py --exp-dir $exp_dir
    exit 0
fi
