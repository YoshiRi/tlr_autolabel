#!/usr/bin/env python3
"""Overlay Tier B boxes on a camera's frames and encode a review MP4.

Color by source_type: matched detection = green, unmatched detection = orange,
interpolated (gap-filled) = cyan, map-presence (detector-missed) = magenta. Label with state; interpolated marked [I].
"""
import json, os, subprocess, sys
from collections import defaultdict
import cv2

DS = os.path.expanduser("~/.webauto/data/data/annotation_dataset/c1af6a38-62db-468f-a3dc-30e88cfe8c92/0")
CAM = sys.argv[1] if len(sys.argv) > 1 else "CAM_FRONT"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/claude-1000/-home-yoshiri-autoware-src-tools-autoware-ml-model-launchers/496e5861-1b6d-462b-b13a-9834d7c10432/scratchpad/review_%s.mp4" % CAM
W = 1280
FPS = 12

anns = json.load(open(f"{DS}/annotation/traffic_signal_2d_ann.json"))["annotations"]
by_file = defaultdict(list)
for a in anns:
    if a["channel"] == CAM:
        by_file[os.path.basename(a["filename"])].append(a)

files = sorted(os.listdir(f"{DS}/data/{CAM}"))
h0, w0 = cv2.imread(f"{DS}/data/{CAM}/{files[0]}").shape[:2]
scale = W / w0
H = int(h0 * scale)

tmp = os.path.dirname(OUT) + f"/_frames_{CAM}"
os.makedirs(tmp, exist_ok=True)
COL = {"matched": (0, 200, 0), "unmatched": (0, 140, 255),
       "interpolated": (255, 200, 0), "map_presence": (255, 0, 255)}

for idx, fn in enumerate(files):
    img = cv2.imread(f"{DS}/data/{CAM}/{fn}")
    img = cv2.resize(img, (W, H))
    counts = defaultdict(int)
    for a in by_file.get(fn, []):
        at = a["attributes"]
        st = at["source_type"]
        if st == "interpolated":
            kind = "interpolated"
        elif st == "map_presence":
            kind = "map_presence"
        elif at.get("map_traffic_light_id"):
            kind = "matched"
        else:
            kind = "unmatched"
        counts[kind] += 1
        x0, y0, x1, y1 = [int(v * scale) for v in a["box2d"]]
        c = COL[kind]
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 2)
        pfx = {"interpolated":"[I]","map_presence":"[M]"}.get(kind,"")
        tag = pfx + (at["state"][:14])
        ty = y0 - 5 if y0 > 14 else y1 + 14
        cv2.putText(img, tag, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    # header: frame + legend + counts
    cv2.rectangle(img, (0, 0), (W, 22), (30, 30, 30), -1)
    cv2.putText(img, f"{CAM} {fn}  matched={counts['matched']} "
                f"interp={counts['interpolated']} mapfill={counts['map_presence']} unmatched={counts['unmatched']}",
                (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (k, c) in enumerate(COL.items()):
        cv2.putText(img, k, (W - 360 + i * 120, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    cv2.imwrite(f"{tmp}/{idx:05d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])

subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", f"{tmp}/%05d.jpg",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", OUT],
               check=True, stderr=subprocess.DEVNULL)
for f in os.listdir(tmp):
    os.remove(f"{tmp}/{f}")
os.rmdir(tmp)
print("wrote", OUT, "size", round(os.path.getsize(OUT) / 1e6, 1), "MB")
