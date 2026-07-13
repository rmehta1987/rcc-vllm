#!/bin/bash
#SBATCH --job-name=hf-stage-glm52
#SBATCH --partition=build
#SBATCH --account=rcc-staff
#SBATCH --qos=build
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/rcc/mehta5/vllm/models/.stage-glm52-%j.log

# Resume the download of zai-org/GLM-5.2-FP8 (755.7 GB, 150 files, 141 shards)
# into models/GLM-5.2-FP8. The 2026-07-09 attempt hung for ~20 h with no bytes
# written: the transfer went through the xet backend, which has no stall
# timeout. Disable xet and set a read timeout so a dead connection fails fast
# and the retry loop resumes from the .incomplete chunks instead of hanging.

set -u

export HF_HOME=/project/rcc/mehta5/hf-cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60

HF=/project/rcc/mehta5/conda-envs/vllm-probe/bin/hf
PY=/project/rcc/mehta5/conda-envs/vllm-probe/bin/python
DEST=/project/rcc/mehta5/vllm/models/GLM-5.2-FP8

rc=1
for attempt in 1 2 3 4 5; do
  echo "=== $(date '+%F %T') attempt $attempt: zai-org/GLM-5.2-FP8 -> $DEST ==="
  "$HF" download zai-org/GLM-5.2-FP8 --local-dir "$DEST" --max-workers 8
  rc=$?
  [ "$rc" -eq 0 ] && break
  echo "=== $(date '+%F %T') attempt $attempt failed (rc=$rc); retrying in 30 s ==="
  sleep 30
done
echo "=== $(date '+%F %T') download loop finished (rc=$rc) ==="

# Verify every shard named in the safetensors index is present and no
# .incomplete chunks remain.
"$PY" - <<'EOF'
import glob, json, os, sys
dest = "/project/rcc/mehta5/vllm/models/GLM-5.2-FP8"
idx_path = os.path.join(dest, "model.safetensors.index.json")
if not os.path.exists(idx_path):
    print("VERIFY FAILED: no model.safetensors.index.json")
    sys.exit(1)
idx = json.load(open(idx_path))
shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(dest, s))]
leftover = glob.glob(os.path.join(dest, ".cache/huggingface/download/*.incomplete"))
print(f"shards: {len(shards)} expected, {len(shards) - len(missing)} present; "
      f"{len(leftover)} incomplete chunks")
if missing or leftover:
    print("MISSING:", missing[:10])
    sys.exit(1)
print("VERIFY OK")
EOF
