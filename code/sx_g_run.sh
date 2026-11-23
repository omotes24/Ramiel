#!/bin/bash
# sx_g_run.sh -- PROTOCOL_SX.md, addendum G: extraction for the seeds given (bank on four GPUs, then the test sets), tier after tier.
#   A bank part that is already complete (meta_bank_p<i>of4.json) is not repeated; a part that is running is waited for.
cd ~/ramiel; PY=${PY:-python}
for seed in "$@"; do for tag in lsb256 adm256c; do
  D=runs/sx/g/feat/s$seed/$tag
  for g in 0 1 2 3; do
    if [ ! -f $D/meta_bank_p${g}of4.json ] && ! pgrep -f "sx_g_extract.py $g $tag $seed bank $g/4" > /dev/null; then
      $PY code/sx_g_extract.py $g $tag $seed bank $g/4 > logs/sx/g_${tag}_s${seed}_bank$g.log 2>&1 &
    fi
  done
  wait
  while pgrep -f "sx_g_extract.py [0-3] $tag $seed bank" > /dev/null; do sleep 10; done
  for g in 0 1 2 3; do [ -f $D/meta_bank_p${g}of4.json ] || { echo "SX_G_RUN_FAILED $tag $seed bank $g"; exit 1; }; done
  [ -f $D/meta.json ] || $PY code/sx_g_extract.py 0 $tag $seed sets > logs/sx/g_${tag}_s${seed}_sets.log 2>&1
  [ -f $D/meta.json ] || { echo "SX_G_RUN_FAILED $tag $seed sets"; exit 1; }
  echo "SX_G_RUN_TIER_DONE $tag $seed $(date -u +%FT%TZ)"
done; echo "SX_G_RUN_SEED_DONE $seed $(date -u +%FT%TZ)"; done
