#!/bin/bash
# CPU-only metric pass: everything except the MLIP-backed stages.
# Runs on copies so it cannot collide with the in-flight MLIP metric jobs.
cd /scratch/yw10199/CrystalReasoner
export TMPDIR=/scratch/yw10199/tmp
for src in repro/out/*.parquet; do
  tag=$(basename "$src" .parquet)
  [ "$tag" = "smoke_metrics" ] && continue
  dst="repro/nomlip/$tag.parquet"
  if [ -f "$dst" ]; then echo "[skip] $tag"; continue; fi
  cp "$src" "$dst.tmp" || continue
  mv "$dst.tmp" "$dst"
  echo "=== $tag ==="
  .venv/bin/python -m crysreas.metric_process \
    --path "$dst" \
    --metrics-name simple_structure structure_validity smact_validity \
                   composition_consistency spacegroup_consistency \
    --prompt-type conditional+thinking \
    --num-workers 8 --forced
  echo "=== $tag rc=$? ==="
done
echo "NOMLIP DONE $(date)"
