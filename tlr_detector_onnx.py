#!/usr/bin/env python3
"""TLR YOLOX detector inference via onnxruntime (recovered from session 3778405d, 2026-07-10).

Decodes the yolox-sPlus traffic_light detector ONNX output (18900x6, num_class=1:
4 box + 1 obj + 1 class) into detections. Grid/stride decode matches the Autoware
tensorrt_yolox traffic_light detector node.
"""
import onnxruntime as ort
import numpy as np
import cv2

img_path = "/home/yoshiri/Downloads/tlr_sample.png"
model_path = "/home/yoshiri/autoware_data/TLRtest/traffic_light_detector/yolox-sPlus-opt-Co_MLOps-traffic_light-960x960_20260629_best.onnx"

img = cv2.imread(img_path)
H, W = 960, 960
# Letterbox exactly like autoware_tensorrt_yolox: keep aspect ratio, pad 114,
# BGR (no swap), norm_factor=1.0 (no /255). Boxes decode in the WxH net space;
# map back to the original image by dividing by `scale`.
ih, iw = img.shape[:2]
scale = min(W / iw, H / ih)
nw, nh = int(iw * scale), int(ih * scale)
canvas = np.full((H, W, 3), 114, dtype=np.uint8)
canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
blob = canvas.astype(np.float32).transpose(2, 0, 1)[None, ...]  # BGR, NCHW, 0-255

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
out = sess.run(None, {"images": blob})[0][0]  # (18900, 6)

strides = [8, 16, 32]
grids = []
for s in strides:
    gw, gh = W // s, H // s
    for g1 in range(gh):
        for g0 in range(gw):
            grids.append((g0, g1, s))
assert len(grids) == out.shape[0], (len(grids), out.shape[0])

score_thr = 0.35
dets = []
for idx, (g0, g1, s) in enumerate(grids):
    obj = out[idx, 4]
    cls = out[idx, 5]
    prob = obj * cls
    if prob > score_thr:
        cx = (out[idx, 0] + g0) * s
        cy = (out[idx, 1] + g1) * s
        w = np.exp(out[idx, 2]) * s
        h = np.exp(out[idx, 3]) * s
        dets.append((prob, cx - w/2, cy - h/2, cx + w/2, cy + h/2, w, h))

dets.sort(key=lambda d: -d[0])
print(f"total candidate detections above {score_thr}: {len(dets)}")
for prob, x0, y0, x1, y1, w, h in dets[:15]:
    print(f"score={prob:.3f} box=({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f}) w={w:.1f} h={h:.1f}")
