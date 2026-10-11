#!/bin/bash
# Download and extract SubT-MRS SmokeRoom (folder release + reference trajectory + calibration)
# from the official Google Drive links on superodometry.com (ICCV'23 SLAM Challenge).
#   sbatch scripts/download_subtmrs_smokeroom.sbatch [DEST_ROOT]
# Result: DEST_ROOT/SubT_MRS_Hawkins_Smoke_Handheld/{SmokeRoom_Folder,SubT_MRS_Smoke_Room,*.yaml}
# ~8.6 GB download, ~9.4 GB extracted. Large downloads are killed on PACE login nodes.
#SBATCH --job-name=dl-subtmrs-smoke
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=dl-subtmrs-smoke-%j.out
set -euo pipefail
DEST="${1:-$HOME/scratch/datasets/subtmrs}/SubT_MRS_Hawkins_Smoke_Handheld"
mkdir -p "$DEST"
cd "$DEST"

fetch() {  # fetch <google-drive-file-id> <output-name>
    [ -s "$2" ] && { echo "have $2"; return; }
    curl -fL --retry 5 --retry-delay 10 -C - -o "$2.part" \
        "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t"
    mv "$2.part" "$2"
}

fetch 1Q0JiqiIgGZ-7DZKZDNJ68F7rpC2rrLtT SubT_MRS_Hawkins_Smoke_Handheld_Intrinsics.yaml
fetch 1HjWlRVQQvgrFGlgRxcczRt92Xy_P-5Ij SubT_MRS_Hawkins_Smoke_Handheld_Extrinsics.yaml
fetch 1YFm6F-c5NI84zKNqyCWA-_c-KABSKF4y SubT_MRS_Hawkins_Smoke_Room_GT_IMU.zip
fetch 140UJgOCa-2itHUZQfwOK1zISD8XRGw51 SubT_MRS_Hawkins_Smoke_Handheld.zip

for z in SubT_MRS_Hawkins_Smoke_Room_GT_IMU.zip SubT_MRS_Hawkins_Smoke_Handheld.zip; do
    python3 -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$z"
done
sha256sum ./*.zip ./*.yaml > SHA256SUMS
ls SmokeRoom_Folder/cam_0 | wc -l
echo "done: $DEST"
