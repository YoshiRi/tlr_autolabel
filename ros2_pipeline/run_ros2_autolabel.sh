#!/usr/bin/env bash
# Autolabel a directory of images with the LIVE Autoware TLR recognition graph
# (production tensorrt_yolox detector + traffic_light_classifier, int8 TensorRT).
#
# Runs: graph (tlr_detect_and_classifier.launch.xml) + collector + feeder, then
# tears everything down. First call builds the TensorRT engines (slow, once).
#
# Usage:
#   run_ros2_autolabel.sh <image_dir> <out_dir> [camera_ns] [rate] [limit] [image_root]
# image_root: make the JSON `image` field relative to this root, to match the
#   offline run's --image-root (needed for a clean L5 parity comparison).
# NOTE: no `set -u` -- ROS 2 setup.bash references unbound vars and would abort.
set -o pipefail

IMG_DIR="${1:?image_dir}"; OUT_DIR="${2:?out_dir}"
CAM="${3:-autolabel}"; RATE="${4:-3}"; LIMIT="${5:-0}"; IMG_ROOT="${6:-}"
RUN_ID="ros2-$(date +%Y%m%d-%H%M%S)"
WS=/home/yoshiri/autoware
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG_TOPIC="/tlr_autolabel/image"
BASE="/perception/traffic_light_recognition/${CAM}"
ROIS="${BASE}/detection/rois"
CAR="${BASE}/classification/car/traffic_signals"
PED="${BASE}/classification/pedestrian/traffic_signals"
MAP="/tmp/tlr_frame_map_${CAM}.json"
mkdir -p "$OUT_DIR"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

# Detector override (for L5 parity: use the SAME detector as the offline run).
# Default = the 1280 single-class TLRtest engine that the offline pipeline uses,
# so parity measures implementation (letterbox/decode/int8) not model choice.
# Set DET_PKG_PATH="" to fall back to the launch default (960 car+ped detector).
DET_PKG_PATH="${DET_PKG_PATH-/home/yoshiri/autoware_data/TLRtest/traffic_light_detector}"
DET_PARAM="${DET_PARAM-${DET_PKG_PATH:+$DET_PKG_PATH/ml_package_yolox.1280x1280_20260703_best.param.yaml}}"

LAUNCH_ARGS=(
  autoware_ml_model_launchers tlr_detect_and_classifier.launch.xml
  camera_namespace:="$CAM"
  input/image:="$IMG_TOPIC"
  use_decompress:=false
  use_sim_time:=false
  enable_classification:=true
  car_classifier_type:=2
  car_classifier_model_name:=traffic_light_lamp_recognizer_comlops.onnx
  pedestrian_classifier_type:=1
)
[ -n "$DET_PKG_PATH" ] && LAUNCH_ARGS+=(detector_ml_package_path:="$DET_PKG_PATH")
[ -n "$DET_PARAM" ] && LAUNCH_ARGS+=(detector_param_path:="$DET_PARAM")
# Note: car uses LampRecognizer (YOLOX classifier, type 2, lamp_recognizer onnx);
# pedestrian stays on the CNN model (type 1, prebuilt engine). Pointing both at
# the same lamp onnx makes them race to build the SAME engine file on one GPU.

# 1) build TensorRT engines once (nodes exit after building)
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "[run] building engines (build_only) ..."
  ros2 launch "${LAUNCH_ARGS[@]}" build_only:=true 2>&1 | sed 's/^/[build] /'
fi

# 2) launch the live graph
echo "[run] launching graph ..."
ros2 launch "${LAUNCH_ARGS[@]}" > "$OUT_DIR/_graph.log" 2>&1 &
GRAPH_PID=$!
sleep 12   # let nodes come up & load engines

# 3) collector (must subscribe so the lazy classifier runs)
echo "[run] starting collector ..."
python3 "$HERE/tlr_label_collector.py" --out-dir "$OUT_DIR" --frame-map "$MAP" \
  --rois-topic "$ROIS" --car-topic "$CAR" --ped-topic "$PED" \
  --run-id "$RUN_ID" ${IMG_ROOT:+--image-root "$IMG_ROOT"} \
  > "$OUT_DIR/_collector.log" 2>&1 &
COLL_PID=$!
sleep 3

# 4) feed images (blocks until done + grace)
echo "[run] feeding images ..."
python3 "$HERE/tlr_image_feeder.py" --image-dir "$IMG_DIR" --topic "$IMG_TOPIC" \
  --rate "$RATE" --frame-id "$CAM" --map-out "$MAP" --limit "$LIMIT" \
  2>&1 | sed 's/^/[feed] /'

echo "[run] feeding done; draining collector ..."
sleep 8
kill -INT "$COLL_PID" 2>/dev/null; wait "$COLL_PID" 2>/dev/null
kill -INT "$GRAPH_PID" 2>/dev/null; wait "$GRAPH_PID" 2>/dev/null
echo "[run] done. JSON in $OUT_DIR ($(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l) files)"
