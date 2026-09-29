#!/usr/bin/bash
#SBATCH --job-name=PRE-ACT_test
#SBATCH --account=TOBEFILLED
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=TOBEFILLED
#SBATCH --output=logs/%A-%x.out
#SBATCH --error=logs/%A-%x.out
#SBATCH --time=0:30:00

source ~/.bashrc
conda activate preact

module purge
module load CUDA/12.8.0
module load GCC/12.3.0

cd /path/to/PRE-ACT
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
#export WANDB_MODE=offline

# Check if 2 arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch run_infer.slurm <model_id> <subset>"
    exit 1
fi

# User inputs
model_id=$1
subset=$2

echo "Running with:"
echo "model_id=${model_id}"
echo "subset=${subset}"

nvidia-smi

srun torchrun \
  --standalone \
  --nproc_per_node=4 \
  test_encoder_anticipation_ddp.py \
  --batch-size 200 \
  --num-workers 1 \
  --checkpoint ./outputs_${subset}/$model_id/best_mauc_0_1.pt \
  --root ../../data/MM_AU \
  --metadata-json ../../data/MM_AU/video_metadata.json \
  --subset ${subset} \
  --split-name test \
  --run-sliding-window \
  --eval-score-key risk_score \
  --eval-anno-json ./annotations/mm_au_cap_anno.json \
  --save-dir ./results_${subset}/${model_id}_best_mauc_${subset}
