#!/bin/bash
# Background model stager for the model-refresh loop.
#
# Xet-disabled `hf download` + retry loop + precise shard-index verify (Gate D),
# run IN the build allocation as a detached background process -- NOT a nested
# build job (the build QOS is one job per user and the orchestrator already holds
# it; session_start.md §0 partition-routing discipline). Mirrors the frozen
# recipes sbatch_stage_glm52.sh / sbatch_stage_qwen35_122b.sh minus the #SBATCH
# header, and adds an EXACT tensor-byte check against the index metadata.total_size
# (parses each safetensors 8-byte header-length prefix; sum(filesize-8-hdr)==total_size).
#
# Usage:  stage_model_bg.sh <hf_repo_id> <dest_dir>
set -u
REPO="${1:?usage: stage_model_bg.sh <hf_repo_id> <dest_dir>}"
DEST="${2:?usage: stage_model_bg.sh <hf_repo_id> <dest_dir>}"

export HF_HOME=/project/rcc/mehta5/hf-cache
export HF_HUB_DISABLE_XET=1          # xet backend has no stall timeout -> hangs
export HF_HUB_DOWNLOAD_TIMEOUT=60    # fail a dead connection fast so retry resumes

HF=/project/rcc/mehta5/conda-envs/vllm-probe/bin/hf
PY=/project/rcc/mehta5/conda-envs/vllm-probe/bin/python

rc=1
for attempt in 1 2 3 4 5 6 7 8; do
  echo "=== $(date '+%F %T') attempt $attempt: $REPO -> $DEST ==="
  "$HF" download "$REPO" --local-dir "$DEST" --max-workers 8
  rc=$?
  [ "$rc" -eq 0 ] && break
  echo "=== $(date '+%F %T') attempt $attempt failed (rc=$rc); retrying in 30 s ==="
  sleep 30
done
echo "=== $(date '+%F %T') download loop finished (rc=$rc) ==="

# --- Gate D: shards present + non-zero, no .incomplete chunks, tensor bytes exact ---
"$PY" - "$DEST" <<'EOF'
import glob, json, os, sys
dest = sys.argv[1]
idx_path = os.path.join(dest, "model.safetensors.index.json")
single   = os.path.join(dest, "model.safetensors")

def tensor_bytes(path):
    # safetensors layout: 8-byte little-endian header length, then JSON header,
    # then raw tensor data. tensor bytes = filesize - 8 - header_len.
    with open(path, "rb") as f:
        hlen = int.from_bytes(f.read(8), "little")
    return os.path.getsize(path) - 8 - hlen

if os.path.exists(idx_path):
    idx = json.load(open(idx_path))
    shards = sorted(set(idx["weight_map"].values()))
    total_size = idx.get("metadata", {}).get("total_size")
elif os.path.exists(single):
    shards, total_size = ["model.safetensors"], None
else:
    print("GATE-D FAIL: no model.safetensors.index.json and no model.safetensors")
    sys.exit(1)

missing  = [s for s in shards
           if not os.path.exists(os.path.join(dest, s))
           or os.path.getsize(os.path.join(dest, s)) == 0]
leftover = glob.glob(os.path.join(dest, ".cache/huggingface/download/*.incomplete"))
tbytes   = sum(tensor_bytes(os.path.join(dest, s))
               for s in shards if os.path.exists(os.path.join(dest, s)))
print(f"shards: {len(shards)} expected, {len(shards)-len(missing)} present nonzero; "
      f"{len(leftover)} incomplete; tensor_bytes={tbytes} index_total_size={total_size}")
if missing or leftover:
    print("GATE-D FAIL missing/empty:", missing[:10]); sys.exit(1)
if total_size is not None and tbytes != total_size:
    print(f"GATE-D FAIL: tensor bytes {tbytes} != index total_size {total_size}"); sys.exit(1)
print("GATE-D PASS")
EOF
gate=$?
echo "=== $(date '+%F %T') stager exit (download rc=$rc, gate-d rc=$gate) ==="
exit $gate
