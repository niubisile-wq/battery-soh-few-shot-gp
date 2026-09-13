#!/usr/bin/env bash
set -u

dataset_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

fetch() {
    local url="$1"
    local output="$2"
    local log_file="${output}.download.log"

    mkdir -p "$(dirname "$output")"
    printf 'START\t%s\n' "$output"
    if wget \
        --continue \
        --retry-connrefused \
        --waitretry=3 \
        --read-timeout=90 \
        --timeout=90 \
        --tries=10 \
        --output-document="$output" \
        "$url" >"$log_file" 2>&1; then
        printf 'DONE\t%s\n' "$output"
        return 0
    fi

    printf 'FAILED\t%s\tsee %s\n' "$output" "$log_file" >&2
    return 1
}

export -f fetch

download_matr() {
    fetch \
        'https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download' \
        "$dataset_root/MATR/2017-05-12_batchdata_updated_struct_errorcorrect.mat" &
    fetch \
        'https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download' \
        "$dataset_root/MATR/2017-06-30_batchdata_updated_struct_errorcorrect.mat" &
    fetch \
        'https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download' \
        "$dataset_root/MATR/2018-04-12_batchdata_updated_struct_errorcorrect.mat" &
    wait
}

download_calce() {
    local active=0
    local failed=0
    local relative_path
    local url

    while IFS=$'\t' read -r relative_path url; do
        fetch "$url" "$dataset_root/CALCE/$relative_path" &
        active=$((active + 1))
        if (( active >= 3 )); then
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

download_nasa() {
    fetch \
        'https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip' \
        "$dataset_root/NASA/NASA_PCoE_Battery_Data_Set.zip"
}

download_oxford() {
    fetch \
        'https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/m5ac36a1e2073852e4f1f7dee647909a7' \
        "$dataset_root/Oxford_Battery_Degradation_Dataset_1/Oxford_Battery_Degradation_Dataset_1.mat" &
    fetch \
        'https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/m43cc05e7c5f1245f4895d9dbd495e52f' \
        "$dataset_root/Oxford_Battery_Degradation_Dataset_1/Readme.txt" &
    fetch \
        'https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/me9fc40a60ac98708f1b73f3a836548e9' \
        "$dataset_root/Oxford_Battery_Degradation_Dataset_1/ExampleDC_C1.mat" &
    wait
}

download_matr &
pid_matr=$!
download_calce &
pid_calce=$!
download_nasa &
pid_nasa=$!
download_oxford &
pid_oxford=$!

status=0
for pid in "$pid_matr" "$pid_calce" "$pid_nasa" "$pid_oxford"; do
    wait "$pid" || status=1
done

exit "$status"
