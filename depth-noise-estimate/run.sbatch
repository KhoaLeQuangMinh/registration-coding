#!/bin/bash
# Depth Anything V2 on SERV-CT, then the shape-error analysis. Submit from this folder:  sbatch run.sbatch
#SBATCH --job-name=depth-noise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/khoalq/registration/depth-noise-estimate/results/slurm-%j.log
set -euo pipefail

ROOT=/data/khoalq/registration/depth-noise-estimate
PY=/data/khoalq/miniconda3/envs/reg/bin/python
export HF_HOME=/data/khoalq/registration/tmp/hf          # model weights on /data, never in /home
cd "$SLURM_SUBMIT_DIR"

$PY predict.py --data "$ROOT/data/SERV-CT" --out "$ROOT/results/predictions"
$PY analyse.py --data "$ROOT/data/SERV-CT" --pred "$ROOT/results/predictions" --out "$ROOT/results"
