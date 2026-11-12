#!/bin/bash
# Expand SP_SLAM3's quicklz-compressed DBoW3 vocabulary into an uncompressed DBoW3 stream that vpr/dbow.py reads.
# Builds DBoW3's own quicklz.c (copied out of the SP_SLAM3 image) with the host gcc.
#   scripts/dbow3_decompress.sh [sif] [out]
# defaults: spslam3_9439971_r6.sif -> /storage/ice1/3/6/bkuang7/slam-bench-work/vpr_weights/superpoint_voc.dbow3
set -euo pipefail
export PATH=/storage/ice1/3/6/bkuang7/install-dir/bin:$PATH
W=/storage/ice1/3/6/bkuang7/slam-bench-work
SIF=${1:-$W/containers/spslam3_9439971_r6.sif}
OUT=${2:-$W/vpr_weights/superpoint_voc.dbow3}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
for f in quicklz.c quicklz.h; do apptainer exec "$SIF" cat /opt/SP_SLAM3/Thirdparty/DBoW3/src/$f > "$T/$f"; done
apptainer exec "$SIF" cat /opt/SP_SLAM3/Vocabulary/superpoint_voc.dbow3 > "$T/in.dbow3"
cat > "$T/unq.c" <<'EOF'
/* DBoW3 Vocabulary::toStream(compressed) -> toStream(uncompressed): header (magic, flag, n_nodes) + payload */
#include "quicklz.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb"), *o = fopen(argv[2], "wb");
  uint64_t sig; char comp, zero = 0; uint32_t nn, nch;
  if (!f || !o || fread(&sig, 8, 1, f) != 1 || fread(&comp, 1, 1, f) != 1 || fread(&nn, 4, 1, f) != 1) return 1;
  fwrite(&sig, 8, 1, o); fwrite(&zero, 1, 1, o); fwrite(&nn, 4, 1, o);
  if (!comp) { char b[65536]; size_t r; while ((r = fread(b, 1, sizeof b, f)) > 0) fwrite(b, 1, r, o); return 0; }
  if (fread(&nch, 4, 1, f) != 1) return 1;
  qlz_state_decompress *st = calloc(1, sizeof(qlz_state_decompress));
  char *in = malloc(10000 + 400), *out = malloc(10000);  /* DBoW3 chunk size 10000 */
  for (uint32_t i = 0; i < nch; i++) {
    if (fread(in, 1, 9, f) != 9) return 1;
    size_t c = qlz_size_compressed(in);
    if (fread(in + 9, 1, c - 9, f) != c - 9) return 1;
    fwrite(out, 1, qlz_decompress(in, out, st), o);
  }
  return fclose(o);
}
EOF
gcc -O2 -o "$T/unq" "$T/unq.c" "$T/quicklz.c"
"$T/unq" "$T/in.dbow3" "$OUT"
ls -la "$OUT"
