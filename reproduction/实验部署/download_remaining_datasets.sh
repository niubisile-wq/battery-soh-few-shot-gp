#!/usr/bin/env bash
set -u

dataset_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

fetch_direct() {
    local url="$1"
    local output="$2"
    local log_file="${output}.download.log"

    printf 'START-DIRECT\t%s\n' "$output"
    if wget \
        --no-proxy \
        --continue \
        --retry-connrefused \
        --waitretry=2 \
        --read-timeout=90 \
        --timeout=90 \
        --tries=10 \
        --output-document="$output" \
        "$url" >"$log_file" 2>&1; then
        printf 'DONE-DIRECT\t%s\n' "$output"
        return 0
    fi
    printf 'FAILED-DIRECT\t%s\n' "$output" >&2
    return 1
}

fetch_proxy() {
    local url="$1"
    local output="$2"
    local log_file="${output}.download.log"

    printf 'START-PROXY\t%s\n' "$output"
    if wget \
        --continue \
        --retry-connrefused \
        --waitretry=2 \
        --read-timeout=90 \
        --timeout=90 \
        --tries=10 \
        --output-document="$output" \
        "$url" >"$log_file" 2>&1; then
        printf 'DONE-PROXY\t%s\n' "$output"
        return 0
    fi
    printf 'FAILED-PROXY\t%s\n' "$output" >&2
    return 1
}

download_calce() {
    local active=0
    local failed=0
    local relative_path
    local url

    while IFS=$'\t' read -r relative_path url; do
        fetch_direct "$url" "$dataset_root/CALCE/$relative_path" &
        active=$((active + 1))
        if (( active >= 6 )); then
            wait -n || failed=1
            active=$((active - 1))
        fi
    done < "$dataset_root/CALCE/download_urls.tsv"

    while (( active > 0 )); do
        wait -n || failed=1
        active=$((active - 1))
    done
    return "$failed"
}

download_small_proxy_sources() {
    local failed=0

    fetch_proxy \
        'https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/m5ac36a1e2073852e4f1f7dee647909a7' \
        "$dataset_root/Oxford_Battery_Degradation_Dataset_1/Oxford_Battery_Degradation_Dataset_1.mat" &
    pid_oxford=$!

    fetch_proxy \
        'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip' \
        "$dataset_root/NASA/NASA_PCoE_Battery_Data_Set.zip" &
    pid_nasa=$!

    wait "$pid_oxford" || failed=1
    wait "$pid_nasa" || failed=1
    return "$failed"
}

download_calce &
pid_calce=$!
download_small_proxy_sources &
pid_small=$!

status=0
wait "$pid_calce" || status=1
wait "$pid_small" || status=1
exit "$status"
