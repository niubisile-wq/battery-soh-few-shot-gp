#!/usr/bin/env bash
set -u

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
lock_file="实验部署/batterylife_download.lock"
log_file="实验部署/batterylife_download.log"
manifest="数据集/BatteryLife_v12_processed/range_manifest_selected_v12.json"

exec 9>"$lock_file"
flock -n 9 || exit 0

while :; do
    python 数据集/parallel_range_download.py "$manifest" \
        --workers 4 --segment-mib 64 --use-env-proxy >>"$log_file" 2>&1
    status=$?
    if [ "$status" -eq 0 ]; then
        printf '%s\n' "BatteryLife v12 download and assembly complete" >>"$log_file"
        exit 0
    fi
    printf 'Downloader exited with status %s; retrying in 15 seconds\n' "$status" >>"$log_file"
    sleep 15
done
