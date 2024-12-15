#!/bin/bash

#SBATCH --job-name=llmcompr

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=<mail>

#SBATCH --output=zh_logs/gen.txt
#SBATCH --time=0-05

#SBATCH --mem=100G

#SBATCH --nodes=1

#SBATCH -c 8

#SBATCH --gpus=1

srun singularity exec --bind /trinity/home/user/:/home -f --nv /trinity/home/user/images/trl.sif bash -c '
    export HF_HOME=/home/cache/;
    export HF_TOKEN=<token>;
    export SAVING_DIR=/home/cache/;
    export WANDB_API_KEY=<key>;
    ls;
    cd /home/user/;
    nvidia-smi;
    pip list;
    CUDA_LAUNCH_BLOCKING=1;
    python generate_tqa.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --output_path /home/user/no_context.pickle \
    --data_path /home/user/tqa_with_source \
    --cache_dir /home/cache/ \
    --batch_size 8;
'

