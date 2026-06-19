#!/bin/bash
#SBATCH -J HF_Scope_Benchmark
#SBATCH -A hartvigsen_lab
#SBATCH -p gpu
#SBATCH --array=0-0                  # Adjusted for the single model listed below; revert to 0-8 if adding more
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/scope_%A_%a.out
#SBATCH -e logs/scope_%A_%a.err
#SBATCH --requeue

set -euo pipefail

RUN_TAG="scope_50k"

# Environment Setup
module purge
module load gcc/11.4.0
module load miniforge
conda activate <env>

# Dynamic Paths (Sanitized for Public Repo)
export USER=$(whoami)
export SCRATCH_ROOT="/sfs/weka/scratch/$USER"
export REPO_DIR="$(pwd)" # Assumes you submit the job from the root of your repo

export HF_HOME="$SCRATCH_ROOT/hf_home"
export TRANSFORMERS_CACHE="$SCRATCH_ROOT/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH_ROOT/hf_datasets"
export HF_MODULES_CACHE="$SCRATCH_ROOT/hf_modules"
export PYTHONPYCACHEPREFIX="$SCRATCH_ROOT/py_cache"
export TMPDIR="$SCRATCH_ROOT/tmp"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
         "$HF_MODULES_CACHE" "$PYTHONPYCACHEPREFIX" "$TMPDIR"

OUTDIR="$SCRATCH_ROOT/wikidata_scope_results/$RUN_TAG"
mkdir -p "$OUTDIR"

# Data path - expected to be in your scratch space
DATA="$SCRATCH_ROOT/merged_scope_benchmark.jsonl"

MODELS=(
  "nvidia/Nemotron-Cascade-8B"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
SAFE_NAME=$(echo "$MODEL" | tr '/' '_')
OUTFILE="$OUTDIR/${SAFE_NAME}_50k_1.json"

# Update job name dynamically for monitoring
scontrol update JobId=$SLURM_JOB_ID Name="❝ Scope · $SAFE_NAME ❞"

echo "======================================"
echo "🧠 Model: $MODEL"
echo "📄 Output: $OUTFILE"
echo "======================================"

if [[ -f "$OUTFILE" ]]; then
  echo "✅ Output exists, skipping."
  exit 0
fi

# Executing using the updated, relative repository structure
python "$REPO_DIR/source/hf_scope_benchmark.py" \
  --data "$DATA" \
  --model "$MODEL" \
  --out "$OUTFILE" \
  --max_chains 50000 \
  --self_max_chains 10000 \
  --batch_size 16 \
  --max_new_tokens 24 \
  --dtype bfloat16 \
  --scope_turns_only \
  --match_mode relaxed

echo "✅ Finished $MODEL"
