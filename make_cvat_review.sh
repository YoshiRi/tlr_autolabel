#!/usr/bin/env bash
# One command to produce CVAT review task zip(s) from existing autolabels.
#
#   ./make_cvat_review.sh <dataset_root> [camera ...]
#
# Examples:
#   ./make_cvat_review.sh ~/.webauto/.../c1af6a38-.../0                 # all cameras
#   ./make_cvat_review.sh ~/.webauto/.../c1af6a38-.../0 CAM_FRONT       # one camera
#   REFRESH=1 ./make_cvat_review.sh <dataset>                          # re-run L3 first
#   MIN_PRIORITY=1.5 ./make_cvat_review.sh <dataset> CAM_FRONT          # suspicious only
#
# Output: <dataset>/build/cvat_signal/<CAMERA>_*.zip
#   (a "CVAT for images 1.1" dataset: images/ + annotations.xml, incl. the label
#    schema in annotations.xml <meta>).
#
# In CVAT (robust two-step):
#   1. Create a new task. Define the label `traffic_light` with the attribute set
#      from docs/cvat_interop.md (or paste the label JSON in the Raw editor).
#      Upload the zip as the task Data — CVAT extracts images/ as the frames.
#   2. Open the task -> Menu -> Upload annotations -> format "CVAT 1.1" ->
#      pick annotations.xml (from the same zip). Boxes appear on the frames.
#   Then filter `review_priority > 1.5` to jump to suspicious boxes; edit `state`
#   and set `review_status` (accepted/fixed/rejected) per box.
#
# No inference here (fast): reads the existing tlr_autolabel/ labels. Set
# REFRESH=1 to regenerate the Tier B sidecar + RE report first.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
  echo "usage: $0 <dataset_root> [camera ...]" >&2
  exit 1
fi
DS="$(cd "$1" && pwd)"; shift

PY=python3
[ -x "$HERE/run_gpu.sh" ] && command -v "${TLR_GPU_VENV:-$HOME/.venvs/tlr_onnxgpu}/bin/python" >/dev/null 2>&1 \
  && PY="${TLR_GPU_VENV:-$HOME/.venvs/tlr_onnxgpu}/bin/python" || true

if [ "${REFRESH:-0}" = "1" ]; then
  echo ">> refreshing Tier B sidecar + RE report"
  "$PY" "$HERE/match_traffic_lights.py" --dataset-root "$DS"
  "$PY" "$HERE/aggregate_regulatory_signals.py" --dataset-root "$DS"
fi

# cameras: args, or every data/CAM_* dir with jpgs
if [ $# -gt 0 ]; then
  CAMERAS=("$@")
else
  CAMERAS=()
  for d in "$DS"/data/*/; do
    name="$(basename "$d")"
    if compgen -G "$d*.jpg" >/dev/null; then CAMERAS+=("$name"); fi
  done
fi

EXTRA=()
[ -n "${MIN_PRIORITY:-}" ] && EXTRA+=(--min-priority "$MIN_PRIORITY")

echo ">> exporting CVAT tasks for: ${CAMERAS[*]}"
for cam in "${CAMERAS[@]}"; do
  "$PY" "$HERE/export_cvat_signal_task.py" --dataset-root "$DS" \
    --camera "$cam" --count 0 "${EXTRA[@]}"
done

echo
echo "done. Zips in: $DS/build/cvat_signal/"
echo "CVAT: new task with the traffic_light label set (docs/cvat_interop.md),"
echo "  upload the zip as data (images), then Upload annotations -> CVAT 1.1 -> annotations.xml."
echo "  Filter 'review_priority > 1.5' for suspicious boxes; set review_status per box."
echo "After review, export CVAT 1.1 XML and run:"
echo "  python3 import_cvat_signal_annotations.py <exported.xml> --dataset-root $DS \\"
echo "    --output annotation/traffic_signal_2d_ann.json"
