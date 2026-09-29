#!/usr/bin/bash
#SBATCH --job-name=PRE_ACT_Nexar_train
#SBATCH --account=TOBEFILLED
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=TOBEFILLED
#SBATCH --output=logs/%A-%x.out
#SBATCH --error=logs/%A-%x.out
#SBATCH --time=10:00:00

source ~/.bashrc
conda activate preact

module purge
module load CUDA/12.8.0
module load GCC/12.3.0

cd /path/to/PRE-ACT

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

set -x

srun torchrun \
    --standalone \
    --nproc_per_node=4 \
    train_autoencoder_anticipation_ddp.py \
    --num-workers 8 \
        --batch-size 32 \
        --root ../../data/Nexar \
        --subset Nexar \
        --backbone-type videomae \
        --backbone-name MCG-NJU/videomae-large \
        --freeze-backbone \
        --unfreeze-last-n-blocks 4 \
        --backbone-lr 2e-5 \
        --head-lr 2e-4 \
        --lambda-bce 1.0 \
        --lambda-prog 10.0 \
        --lambda-pref 0.1 \
        --custom_risk_mode "exp_above" \
        --progress-alpha 3.0 \
        --anticipation-horizon-sec 2.0 \
        --snippet-len 5 \
        --train-stride 5 \
        --image-size 224 \
        --epochs 30 \
        --eval-every-n-epochs 1 \
        --eval-mode sliding_window \
        --eval-score-key score \
        --eval-anno-json ./annotations/nexar_anno.json \
        --eval-batch-size 128 \
        --save-eval-predictions \
        --pref-loss-type "margin_ranking" \
        --seed 42 \
        --output-dir ./outputs_Nexar/pre_act_videmoae
