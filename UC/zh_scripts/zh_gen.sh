#!/bin/bash

#SBATCH --job-name=llmcompr

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=V.Moskvoretskii@skoltech.ru

#SBATCH --output=zh_logs/gen.txt
#SBATCH --time=0-05

#SBATCH --mem=100G

#SBATCH --nodes=1

#SBATCH -c 8

#SBATCH --gpus=1

srun singularity exec --bind /trinity/home/v.moskvoretskii/:/home -f --nv /trinity/home/v.moskvoretskii/images/trl.sif bash -c '
    export HF_HOME=/home/cache/;
    export HF_TOKEN=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;
    export SAVING_DIR=/home/cache/;
    export WANDB_API_KEY=2b740bffb4c588c7274a6e8cf4e39bd56344d492;
    ls;
    cd /home/TrustGen/;
    nvidia-smi;
    pip list;
    CUDA_LAUNCH_BLOCKING=1;
    python generate_tqa.py \
    --model_path VityaVitalich/Llama3-8b \
    --output_path /home/TrustGen/no_context.pickle \
    --data_path /home/TrustGen/tqa_with_source \
    --cache_dir /home/cache/ \
    --batch_size 8;
'

