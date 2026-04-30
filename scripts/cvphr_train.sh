#!/bin/bash
# Function：Bearing-UAV (cvphr) model training.
# Operation：
#   cd /your/path/of/proj/bearinguav
#   chmod +x ./scripts/cvphr_train.sh
#   ./scripts/cvphr_train.sh


set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

# ====================== parameters ======================
# Debug:    1.'37bc'+1+1+1; 2. 96+0/1+1+1. 
# Training: 1. 96+0/1+100+(1~100). 

# If train AirSim city dataset:
rsi_id='ny'  #'ny' AirSim city
block=3
sample=10  #for debug

# If train GoogleEarth city dataset:
# rsi_id='37bc'  #'37bc' for debug; 96 for training
# block=15
# sample=1  #for debug

# sample=100  # for training, 1 or 100 
# epoch=100  # for training, 1-100
epoch=1  #for debug
is_3d=1  # 0=2d: satellite view, 1=3d: U-S cross-view

log_gcth="${gcth^^}"          # NONE
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO_ROOT/log/c4ma"
mkdir -p "$LOG_DIR"
log_file="${LOG_DIR}/train_phr5_d${rsi_id}_s${sample}_3d${is_3d}_e${epoch}_${timestamp}.log"

echo "[Run] repo_root=$REPO_ROOT"
echo "[Run] log_file=$log_file"

nohup /usr/bin/time -v python -m cvphr.train.cvphr_train \
    --is_3d "$is_3d" \
    --rsi_id "$rsi_id" \
    --n_sample "$sample" \
    --n_block "$block" \
    --num_epochs "$epoch" \
    > "$log_file" 2>&1 &

echo "Started! PID=$!"
echo "Log: $log_file"
