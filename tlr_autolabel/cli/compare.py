"""CLIs for the comparison harness: `run_compare` (L1) and `compare_naive` (L6).

Kept as two entrypoints on purpose. Running inference and analysing its output
are different layers with different costs: a matrix run needs the GPU and the
models, an analysis run needs neither and can be repeated with different
thresholds on the labels a matrix run already wrote.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tlr_autolabel.compare import grid as grid_mod
from tlr_autolabel.compare import naive
from tlr_autolabel.compare.matrix import Matrix, load_matrix, run_matrix
from tlr_autolabel.inference.config import list_presets, resolve_config


# ------------------------------------------------------------------ run_compare


def _inline_combos(specs, defaults):
    """`--combo name=preset[,key=value,...]` for a quick run without a YAML."""
    from tlr_autolabel.compare.matrix import Combo
    from tlr_autolabel.inference.config import CONFIG_FIELDS

    combos = []
    for spec in specs:
        name, sep, rest = spec.partition("=")
        if not sep:
            name, rest = spec, spec
        parts = [p for p in rest.split(",") if p]
        preset, overrides = None, dict(defaults)
        for i, part in enumerate(parts):
            if "=" not in part:
                if i:
                    raise SystemExit(f"--combo {spec!r}: only the first item may be a preset")
                preset = part
                continue
            key, _, value = part.partition("=")
            key = key.strip().replace("-", "_")
            if key not in CONFIG_FIELDS:
                raise SystemExit(f"--combo {spec!r}: unknown configuration key {key!r}")
            overrides[key] = _coerce(value.strip())
        combos.append(Combo(name=name, config=resolve_config(preset, overrides)))
    return combos


def _coerce(value):
    low = value.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def run_compare_main():
    from tlr_autolabel.cli.autolabel import add_frame_source_args, frame_source_spec

    ap = argparse.ArgumentParser(
        description="Run several detector/classifier configurations over the same "
                    "frames and write one Tier A label directory per configuration.")
    ap.add_argument("--matrix", default=None,
                    help="comparison matrix YAML (see configs/compare/)")
    ap.add_argument("--combo", action="append", default=[],
                    help="inline configuration: name=preset[,key=value,...]; "
                         f"presets: {', '.join(list_presets())}")
    ap.add_argument("--out", required=True, help="output directory for the run")
    ap.add_argument("image", nargs="?", default=None,
                    help="image file or directory (or use --video/--bag/--t4-dataset)")
    ap.add_argument("--t4-dataset", default=None, help="T4 dataset root")
    ap.add_argument("--image-root", default=None)
    add_frame_source_args(ap)
    ap.add_argument("--frame-format", default="png", choices=["png", "jpg"],
                    help="format for extracted video/bag frames (png: lossless)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="keep label files that already exist (resume)")
    ap.add_argument("--no-model-digest", action="store_true",
                    help="skip the model sha256 (faster start, weaker provenance)")
    ap.add_argument("--compare", action="store_true",
                    help="run the GT-free comparison on the result right away")
    ap.add_argument("--reference", default=None,
                    help="with --compare: configuration to compare the others against")
    ap.add_argument("--iou-thr", type=float, default=naive.DEFAULT_IOU_THR)
    ap.add_argument("--grid-top", type=int, default=12,
                    help="with --compare: render N most-disagreeing frames side by side")
    args = ap.parse_args()

    if args.matrix:
        matrix = load_matrix(args.matrix)
        if any([args.image, args.video, args.bag, args.t4_dataset]):
            matrix.frames = frame_source_spec(args, prefer_t4_source=True)
    else:
        if not args.combo:
            raise SystemExit("pass --matrix <yaml> or at least one --combo")
        matrix = Matrix(frames=frame_source_spec(args, prefer_t4_source=True),
                        combos=_inline_combos(args.combo, {}))
    if not matrix.frames:
        raise SystemExit("no frames: the matrix has no `frames:` block and no input "
                         "flag was given")

    manifest = run_matrix(matrix, args.out, frame_format=args.frame_format,
                          skip_existing=args.skip_existing,
                          model_digest=not args.no_model_digest)
    if args.compare:
        runs = naive.load_runs_from_manifest(os.path.join(args.out, "run_manifest.json"))
        _report(runs, args.out, reference=args.reference, iou_thr=args.iou_thr,
                grid_top=args.grid_top,
                image_root=os.path.join(args.out, manifest["frames_dir"])
                if manifest.get("frames_dir") else None)


# ---------------------------------------------------------------- compare_naive


def _report(runs, out_dir, reference=None, iou_thr=naive.DEFAULT_IOU_THR,
            grid_top=12, image_root=None, with_consensus=True, grid_video=None):
    report = naive.compare(runs, reference=reference, iou_thr=iou_thr,
                           with_consensus=with_consensus)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "compare_naive.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    text = naive.write_markdown(report, out / "compare_naive.md")
    print(text)
    keys = naive.worst_frames(report, limit=grid_top)
    if grid_top and keys:
        written = grid_mod.render_grids(runs, keys, str(out / "grids"),
                                       image_root=image_root)
        print(f"wrote {len(written)} side-by-side grids -> {out / 'grids'}")
    if grid_video:
        order = sorted(set().union(*(set(r.frames) for r in runs)))
        path = grid_mod.render_grid_video(runs, order, grid_video,
                                          image_root=image_root)
        if path:
            print(f"wrote {path}")
    print(f"wrote {json_path} and {out / 'compare_naive.md'}")
    return report


def compare_naive_main():
    ap = argparse.ArgumentParser(
        description="Compare Tier A label directories without GT or a map: what "
                    "each configuration output, where they disagree, how steady "
                    "each one is.")
    ap.add_argument("runs", nargs="*",
                    help="name=/path/to/labels ... (or bare paths)")
    ap.add_argument("--manifest", default=None,
                    help="run_manifest.json from run_compare.py (loads every combo)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: alongside the manifest, "
                         "else ./build/compare)")
    ap.add_argument("--reference", default=None,
                    help="configuration the others are compared against (default: first)")
    ap.add_argument("--iou-thr", type=float, default=naive.DEFAULT_IOU_THR)
    ap.add_argument("--no-consensus", action="store_true",
                    help="skip the majority-vote pseudo reference")
    ap.add_argument("--grid-top", type=int, default=12,
                    help="render the N most-disagreeing frames side by side (0: none)")
    ap.add_argument("--grid-video", default=None,
                    help="also encode every frame as a side-by-side mp4 (needs ffmpeg)")
    ap.add_argument("--image-root", default=None,
                    help="directory the Tier A `image` paths are relative to "
                         "(e.g. an extracted frames/ dir), for the grid rendering")
    args = ap.parse_args()

    if args.manifest:
        runs = naive.load_runs_from_manifest(args.manifest)
        out_dir = args.out or os.path.dirname(os.path.abspath(args.manifest))
        image_root = args.image_root
        if not image_root:
            manifest = json.loads(Path(args.manifest).read_text())
            if manifest.get("frames_dir"):
                image_root = os.path.join(out_dir, manifest["frames_dir"])
    elif args.runs:
        runs = naive.load_runs(args.runs)
        out_dir = args.out or "build/compare"
        image_root = args.image_root
    else:
        raise SystemExit("pass label directories (name=path ...) or --manifest")

    _report(runs, out_dir, reference=args.reference, iou_thr=args.iou_thr,
            grid_top=args.grid_top, image_root=image_root,
            with_consensus=not args.no_consensus, grid_video=args.grid_video)
