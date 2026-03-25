#!/bin/bash
# Run Bearing-Naver navigation test and log output

# operation：
#     cd /your/path/of/proj/bearinguav
#     chmod +x scripts/run_nav.sh
#     ./scripts/run_nav.sh

LOG_DIR="log/nav"
mkdir -p "$LOG_DIR" || { echo "Error: cannot create $LOG_DIR"; exit 1; }
log_file="${LOG_DIR}/nav_test.log"

nohup python -m naver.runners.nav \
    --uav_2d3d "2d" \
    --cvphr_2d_best_model_dir "./Bearing_UAV/satellite_view" \
    > "$log_file" 2>&1 &

echo "Started in background"
echo "Log: $log_file"
echo "PID: $!"