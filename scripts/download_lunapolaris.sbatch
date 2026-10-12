#!/bin/bash
# Download one LunaPolaris sequence from Zenodo and verify the published md5 checksums.
#   sbatch scripts/download_lunapolaris.sbatch sequence_11 [DEST_ROOT]
# Result: DEST_ROOT/sequence_XX/{metadata.yaml,sequence_XX.db3,...} (~30-36 GB per sequence).
# Parallel HTTP range requests (Zenodo gives ~1-3 MB/s per connection; DL_WORKERS, default 8).
# Re-running resumes partial downloads and skips verified files.
#SBATCH --job-name=dl-lunapolaris
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=dl-lunapolaris-%j.out
set -euo pipefail
SEQ="${1:?usage: download_lunapolaris.sbatch sequence_XX [DEST_ROOT]}"
DEST="${2:-$HOME/scratch/LunaPolaris}/$SEQ"
declare -A RECORDS=(
  [sequence_01]=15737938 [sequence_02]=15744408 [sequence_03]=15744629 [sequence_04]=15745640
  [sequence_05]=15745689 [sequence_06]=15745828 [sequence_07]=15745932 [sequence_08]=15746302
  [sequence_09]=15746894 [sequence_10]=15747096 [sequence_11]=15747348 [sequence_12]=15747458)
REC="${RECORDS[$SEQ]:?unknown sequence $SEQ}"
SCRIPTS="$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" && pwd)/scripts"
mkdir -p "$DEST" && cd "$DEST"
curl -fsSL "https://zenodo.org/api/records/$REC" -o .zenodo_record.json
python3 - <<'PY' > .files.tsv
import json
for f in json.load(open(".zenodo_record.json"))["files"]:
    if f["key"].endswith((".db3", ".yaml", ".txt")):
        print(f["key"], f["checksum"].split(":", 1)[1], f["size"], f["links"]["self"], sep="\t")
PY
while IFS=$'\t' read -r key md5 size url; do
  if [ -f "$key" ] && [ "$(stat -c%s "$key")" = "$size" ] && [ "$(md5sum "$key" | cut -d' ' -f1)" = "$md5" ]; then
    echo "verified $key"; continue
  fi
  echo "downloading $key ($size bytes)"
  python3 "$SCRIPTS/parallel_download.py" "$url" "$key" "$size" --workers "${DL_WORKERS:-8}"
  [ "$(md5sum "$key" | cut -d' ' -f1)" = "$md5" ] || { echo "md5 mismatch for $key" >&2; mv "$key" "$key.bad"; exit 1; }
  echo "verified $key"
done < .files.tsv
[ -s ../calibration_results.json ] || curl -fsSL -o ../calibration_results.json \
  https://zenodo.org/api/records/16728838/files/calibration_results.json/content
echo "done: $DEST"
