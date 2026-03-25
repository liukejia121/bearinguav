#!/bin/bash
# Function：Bearing-UAV (cvphr) model test.
# Operation：
#   cd /your/path/of/proj/bearinguav
#   chmod +x ./scripts/cvphr_test.sh
#   ./scripts/cvphr_test.sh


set -e
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
echo "[Run] project_root=$project_root"
export PYTHONPATH="$project_root:${PYTHONPATH}"

# ====================== parameters ======================
# Debug: '37bc'/96+1+3D model .
# Test:  96 + 100 + 2D/3D/User model.

rsi_id='37bc' # RS image id. '37bc'for debug; 96 for test.
is_3d=1       # 0=2D: satellite view; 1=3D: UAV view.

# Model weight path: 1.2D; 2. 3D; 3. Your trained model.
# 2D model: Pre-trained cvphr_2d_best_model_dir 
# bestpth_dir="./Bearing_UAV/satellite_view" 

# 3D model: Pre-trained cvphr_3d_best_model_dir 
bestpth_dir="./Bearing_UAV/cross_view" 

# User model: Your trained model dir:
# bestpth_dir="${project_root}/results/c4ma/phr5_~~~"

# ====================== log ======================
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${project_root}/log/c4ma"
mkdir -p "$LOG_DIR"

log_file="${LOG_DIR}/test_d${rsi_id}_3d${is_3d}_${timestamp}.log"

echo "[Run] log_file=$log_file"

# ====================== exe ======================
nohup /usr/bin/time -v python -m cvphr.test.cvphr_test \
    --rsi_id "$rsi_id" \
    --is_3d "$is_3d" \
    --bestpth_dir "$bestpth_dir" \
    > "$log_file" 2>&1 &

echo "Started! PID=$!"
echo "Log: $log_file"