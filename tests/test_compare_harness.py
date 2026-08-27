"""End-to-end contract for the comparison harness.

Runs the real matrix runner and the real comparator with the model backends
monkeypatched to fakes (same trick as the CLI smoke test), so the whole path —
matrix parsing, per-combo Tier A output, manifest, agreement metrics, markdown,
grid rendering — is exercised without a GPU or a model file.
"""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tlr_autolabel.compare import grid as grid_mod
from tlr_autolabel.compare import matrix as matrix_mod
from tlr_autolabel.compare import naive
from tlr_autolabel.inference import pipeline as pipeline_mod

# Two configurations that see the world differently: `wide` finds an extra box
# and reads one state differently, `narrow` is the reference.
BOXES = {
    "narrow": [{"prob": 0.9, "x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0}],
    "wide": [{"prob": 0.9, "x1": 11.0, "y1": 11.0, "x2": 31.0, "y2": 31.0},
             {"prob": 0.4, "x1": 60.0, "y1": 60.0, "x2": 80.0, "y2": 80.0}],
}
STATES = {"narrow": "red-circle", "wide": "green-circle"}


class FakeDetector:
    def __init__(self, model_path, _comlops_param, model_type=None):
        self.tag = "wide" if "wide" in str(model_path) else "narrow"
        self.w = self.h = 64
        self.kind = model_type or "yolox"
        self.sess = None

    def detect(self, _img, score_thr):
        return [b for b in BOXES[self.tag] if b["prob"] >= score_thr], 1.0

    def set_keep_classes(self, _names):
        pass


class FakeClassifier:
    def __init__(self, model_path, _param_path, _args, model_type=None):
        self.tag = "wide" if "wide" in str(model_path) else "narrow"
        self.width = self.height = 32
        self.backend = "fake"
        self.kind = model_type or "lamp_recognizer"

    def classify(self, _img, _bbox):
        label = STATES[self.tag]
        color, shape = label.split("-")
        return [{"label": label, "color": color, "shape": shape, "arrow": None,
                 "confidence": 0.9}]


class HarnessTestCase(unittest.TestCase):
    def setUp(self):
        self._orig = (pipeline_mod.Detector, pipeline_mod.LampClassifier)
        pipeline_mod.Detector = FakeDetector
        pipeline_mod.LampClassifier = FakeClassifier
        self.tmp = tempfile.TemporaryDirectory(prefix="tlr_compare_")
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for i in range(4):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[20 + i:40 + i, 20:40] = 255
            cv2.imwrite(str(self.images / f"{i:05d}.png"), img)
        self.models = self.root / "models"
        self.models.mkdir()
        for name in ("narrow_det.onnx", "wide_det.onnx", "narrow_cls.onnx",
                     "wide_cls.onnx", "cls.param.yaml"):
            (self.models / name).write_text("")

    def tearDown(self):
        pipeline_mod.Detector, pipeline_mod.LampClassifier = self._orig
        self.tmp.cleanup()

    def write_matrix(self, extra=""):
        path = self.root / "matrix.yaml"
        path.write_text(f"""
schema_version: tlr_compare_matrix/v1
frames: {{kind: images, path: {self.images}}}
defaults: {{classifier_param: {self.models / 'cls.param.yaml'}}}
combos:
  - name: narrow
    overrides: {{detector: {self.models / 'narrow_det.onnx'},
                 classifier: {self.models / 'narrow_cls.onnx'}}}
  - name: wide
    overrides: {{detector: {self.models / 'wide_det.onnx'},
                 classifier: {self.models / 'wide_cls.onnx'},
                 det_score_thr: 0.3}}
{extra}""")
        return path

    def run_matrix(self):
        out = self.root / "out"
        m = matrix_mod.load_matrix(self.write_matrix())
        manifest = matrix_mod.run_matrix(m, str(out), verbose=False)
        return out, manifest


class MatrixRunTest(HarnessTestCase):
    def test_one_tier_a_directory_per_combo(self):
        out, manifest = self.run_matrix()
        self.assertEqual([c["name"] for c in manifest["combos"]], ["narrow", "wide"])
        for combo in manifest["combos"]:
            self.assertEqual(combo["frames"], 4)
            labels = sorted((out / combo["labels_dir"]).glob("*.json"))
            self.assertEqual([p.name for p in labels],
                             ["00000.json", "00001.json", "00002.json", "00003.json"])
            payload = json.loads(labels[0].read_text())
            self.assertEqual(payload["schema_version"], "tlr_autolabel/v1")
            self.assertIn("timing_ms", payload, "comparison runs record timing")
            self.assertIn("detector_sha256", payload["meta"])
        self.assertEqual(manifest["combos"][0]["signals"], 4)
        self.assertEqual(manifest["combos"][1]["signals"], 8, "wide finds 2 per frame")
        self.assertIsNone(manifest["frames_dir"], "image dirs need no extraction")

    def test_manifest_records_the_full_configuration(self):
        out, manifest = self.run_matrix()
        self.assertTrue((out / "run_manifest.json").exists())
        cfg = manifest["combos"][1]["config"]
        self.assertEqual(cfg["det_score_thr"], 0.3)
        self.assertTrue(cfg["record_timing"])
        self.assertEqual(manifest["frames"]["kind"], "images")

    def test_skip_existing_resumes(self):
        out, _ = self.run_matrix()
        stamp = (out / "labels" / "narrow" / "00000.json").stat().st_mtime_ns
        m = matrix_mod.load_matrix(self.write_matrix())
        manifest = matrix_mod.run_matrix(m, str(out), skip_existing=True, verbose=False)
        self.assertEqual(manifest["combos"][0]["frames"], 0, "everything was skipped")
        self.assertEqual((out / "labels" / "narrow" / "00000.json").stat().st_mtime_ns,
                         stamp, "existing label files are left untouched")


class MatrixParsingTest(HarnessTestCase):
    def _load(self, body):
        path = self.root / "bad.yaml"
        path.write_text(body)
        return matrix_mod.load_matrix(path)

    def test_unknown_combo_key(self):
        with self.assertRaises(SystemExit) as ctx:
            self._load("combos:\n  - {name: a, detector_typo: yolox}\n")
        self.assertIn("unknown key", str(ctx.exception))

    def test_duplicate_names(self):
        with self.assertRaises(SystemExit) as ctx:
            self._load("combos:\n  - {name: a, preset: yolox-960-int8}\n"
                       "  - {name: a, preset: yolox-1920-int8}\n")
        self.assertIn("duplicate combo name", str(ctx.exception))

    def test_empty_combos(self):
        with self.assertRaises(SystemExit) as ctx:
            self._load("combos: []\n")
        self.assertIn("nothing to compare", str(ctx.exception))

    def test_combo_named_after_its_preset_and_defaults_applied(self):
        m = self._load("defaults: {det_score_thr: 0.2}\n"
                       "combos:\n  - {preset: yolox-960-int8}\n")
        self.assertEqual(m.combos[0].name, "yolox-960-int8")
        self.assertEqual(m.combos[0].config.det_score_thr, 0.2)

    def test_shipped_example_matrices_parse(self):
        for path in sorted(Path("configs/compare").glob("*.yaml")):
            m = matrix_mod.load_matrix(path)
            self.assertTrue(m.combos, f"{path}: no combos")
            self.assertFalse(m.frames, f"{path}: frames come from the command line")


class NaiveCompareTest(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.out, self.manifest = self.run_matrix()
        self.runs = naive.load_runs_from_manifest(self.out / "run_manifest.json")

    def test_load_run_keys_by_frame_id(self):
        run = {r.name: r for r in self.runs}["narrow"]
        self.assertEqual(sorted(run.frames), ["00000", "00001", "00002", "00003"])
        self.assertEqual(run.detections, 4)
        self.assertEqual(run.frames["00000"]["signals"][0]["state"], "red-circle")

    def test_pairwise_agreement_and_disagreement_location(self):
        report = naive.compare(self.runs, reference="narrow")
        pair = report["pairwise"][0]
        self.assertEqual(pair["candidate"], "wide")
        self.assertEqual(pair["frames_compared"], 4)
        self.assertEqual(pair["matched"], 4, "the shifted box still matches by IoU")
        self.assertEqual(pair["only_in_reference"], 0)
        self.assertEqual(pair["only_in_candidate"], 4, "wide's extra box each frame")
        self.assertEqual(pair["box_agreement"], 0.5)
        self.assertEqual(pair["state_agreement"], 0.0, "red vs green on every match")
        self.assertEqual(pair["top_state_disagreements"][0],
                         {"reference": "red-circle", "candidate": "green-circle", "n": 4})
        self.assertEqual(pair["only_in_candidate_score"]["p50"], 0.4,
                         "the extra boxes are the low-scoring ones")
        self.assertEqual(len(pair["worst_frames"]), 4)

    def test_per_run_summary(self):
        report = naive.compare(self.runs, reference="narrow")
        wide = report["runs"]["wide"]
        self.assertEqual(wide["detections"], 8)
        self.assertEqual(wide["detections_per_frame"], 2.0)
        self.assertEqual(wide["unknown_rate"], 0.0)
        self.assertIsNotNone(wide["timing_ms"]["total"]["p50"])
        self.assertEqual(wide["meta"]["detector_type"], "yolox")

    def test_stability_sees_a_steady_run_as_steady(self):
        report = naive.compare(self.runs, reference="narrow")
        st = report["runs"]["narrow"]["stability"]
        self.assertEqual(st["state_flip_rate"], 0.0)
        self.assertEqual(st["frame_to_frame_match_rate"], 1.0)
        self.assertEqual(st["tracked_links"], 3, "4 frames -> 3 consecutive pairs")

    def test_consensus_needs_three_configs(self):
        report = naive.compare(self.runs, reference="narrow")
        self.assertFalse(report["consensus"]["available"])
        self.assertIn("at least 3", report["consensus"]["reason"])

    def test_shared_detector_is_flagged(self):
        # both combos point at different files here, so nothing is shared;
        # a run comparing thresholds on one model must report the correlation
        runs = naive.load_runs_from_manifest(self.out / "run_manifest.json")
        runs[1].meta["detector_sha256"] = runs[0].meta["detector_sha256"]
        report = naive.compare(runs, reference=runs[0].name)
        self.assertEqual(len(report["shared_detectors"]), 1)
        self.assertEqual(sorted(report["shared_detectors"][0]["configs"]),
                         ["narrow", "wide"])

    def test_markdown_report_mentions_what_it_cannot_say(self):
        report = naive.compare(self.runs, reference="narrow")
        text = naive.write_markdown(report, self.out / "compare_naive.md")
        self.assertIn("GT-free", text)
        self.assertIn("not which one is right", text)
        self.assertIn("| narrow |", text)
        self.assertIn("| wide |", text)
        self.assertTrue((self.out / "compare_naive.md").exists())

    def test_reference_must_exist(self):
        with self.assertRaises(SystemExit) as ctx:
            naive.compare(self.runs, reference="nope")
        self.assertIn("is not one of", str(ctx.exception))

    def test_grid_rendering_writes_one_panel_image_per_frame(self):
        report = naive.compare(self.runs, reference="narrow")
        keys = naive.worst_frames(report, limit=2)
        self.assertEqual(len(keys), 2)
        written = grid_mod.render_grids(self.runs, keys, str(self.out / "grids"))
        self.assertEqual(len(written), 2)
        img = cv2.imread(written[0])
        self.assertIsNotNone(img)
        self.assertGreater(img.shape[1], 100, "two panels side by side")


class RunCompareCliTest(HarnessTestCase):
    """`run_compare.py` end to end: inline combos, run, compare, grids."""

    def _run_cli(self, argv):
        import sys

        from tlr_autolabel.cli.compare import run_compare_main

        old = sys.argv
        sys.argv = ["scripts/run_compare.py"] + argv
        try:
            run_compare_main()
        finally:
            sys.argv = old

    def test_inline_combos_then_compare(self):
        out = self.root / "cli_out"
        self._run_cli([
            str(self.images), "--out", str(out),
            "--combo", f"narrow=detector={self.models / 'narrow_det.onnx'},"
                       f"classifier={self.models / 'narrow_cls.onnx'},"
                       f"classifier_param={self.models / 'cls.param.yaml'}",
            "--combo", f"wide=detector={self.models / 'wide_det.onnx'},"
                       f"classifier={self.models / 'wide_cls.onnx'},"
                       f"classifier_param={self.models / 'cls.param.yaml'},"
                       f"det_score_thr=0.3",
            "--compare", "--reference", "narrow", "--grid-top", "2",
        ])
        report = json.loads((out / "compare_naive.json").read_text())
        self.assertEqual(report["reference"], "narrow")
        self.assertEqual(report["pairwise"][0]["only_in_candidate"], 4)
        self.assertTrue((out / "compare_naive.md").exists())
        self.assertEqual(len(list((out / "grids").glob("*.grid.png"))), 2)

    def test_combo_rejects_unknown_key(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run_cli([str(self.images), "--out", str(self.root / "x"),
                           "--combo", "a=detector_typo=foo"])
        self.assertIn("unknown configuration key", str(ctx.exception))

    def test_requires_a_matrix_or_a_combo(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run_cli([str(self.images), "--out", str(self.root / "x")])
        self.assertIn("--matrix", str(ctx.exception))

    def test_coerce_scalars(self):
        from tlr_autolabel.cli.compare import _coerce

        self.assertIs(_coerce("true"), True)
        self.assertIs(_coerce("none"), None)
        self.assertEqual(_coerce("128"), 128)
        self.assertEqual(_coerce("0.35"), 0.35)
        self.assertEqual(_coerce("yolox"), "yolox")


class VideoMatrixTest(HarnessTestCase):
    """The video path end to end: extract once, run both configs on the same
    pixels, compare, render grids from the extracted frames."""

    def _write_video(self, path):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                                 10.0, (100, 100))
        if not writer.isOpened():
            self.skipTest("no OpenCV video encoder available")
        for i in range(4):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[20:40, 20 + i:40 + i] = 255
            writer.write(img)
        writer.release()

    def test_frames_are_extracted_once_and_shared(self):
        video = self.root / "clip.avi"
        self._write_video(video)
        path = self.root / "video_matrix.yaml"
        path.write_text(f"""
frames: {{kind: video, uri: {video}}}
defaults: {{classifier_param: {self.models / 'cls.param.yaml'}}}
combos:
  - name: narrow
    overrides: {{detector: {self.models / 'narrow_det.onnx'},
                 classifier: {self.models / 'narrow_cls.onnx'}}}
  - name: wide
    overrides: {{detector: {self.models / 'wide_det.onnx'},
                 classifier: {self.models / 'wide_cls.onnx'}, det_score_thr: 0.3}}
""")
        out = self.root / "video_out"
        manifest = matrix_mod.run_matrix(matrix_mod.load_matrix(path), str(out),
                                        verbose=False)
        self.assertEqual(manifest["frames_dir"], "frames")
        frames_dir = out / "frames"
        self.assertTrue((frames_dir / "frames.json").exists())
        extracted = sorted(p.name for p in frames_dir.glob("*.png"))
        self.assertEqual(extracted, ["000000.png", "000001.png", "000002.png",
                                     "000003.png"])

        payload = json.loads((out / "labels" / "narrow" / "000000.json").read_text())
        self.assertEqual(payload["source"]["kind"], "video")
        self.assertEqual(payload["source"]["origin_ref"], "clip.avi#000000")
        self.assertTrue(payload["image_realpath"].endswith("frames/000000.png"),
                        "a materialized frame is a real file, so review tools work")

        runs = naive.load_runs_from_manifest(out / "run_manifest.json")
        report = naive.compare(runs, reference="narrow")
        self.assertEqual(report["frames"]["common"], 4)
        written = grid_mod.render_grids(runs, naive.worst_frames(report, limit=1),
                                        str(out / "grids"), image_root=str(frames_dir))
        self.assertEqual(len(written), 1)


class SubsampledRunTest(HarnessTestCase):
    """A strided run has no temporal continuity; the report must say so instead
    of showing a 0.0 match rate that reads as flicker."""

    def test_stability_is_marked_not_meaningful(self):
        out, _ = self.run_matrix()
        for name in ("narrow", "wide"):
            for i, path in enumerate(sorted((out / "labels" / name).glob("*.json"))):
                payload = json.loads(path.read_text())
                payload["frame_index"] = i * 100        # as --frame-stride 100 would
                path.write_text(json.dumps(payload))
        runs = naive.load_runs_from_manifest(out / "run_manifest.json")
        report = naive.compare(runs, reference="narrow")
        st = report["runs"]["narrow"]["stability"]
        self.assertFalse(st["meaningful"])
        self.assertEqual(st["median_frame_gap"], 100)
        text = naive.write_markdown(report, out / "sub.md")
        self.assertIn("Stability is `n/a`", text)
        self.assertIn("100 source frames apart", text)


class GridFallbackTest(HarnessTestCase):
    def test_agreeing_runs_still_get_grids(self):
        out, _ = self.run_matrix()
        runs = naive.load_runs_from_manifest(out / "run_manifest.json")
        # make the two runs identical so there is nothing to rank by disagreement
        for rec in runs[1].frames.values():
            rec["signals"] = [dict(s) for s in runs[0].frames[rec["frame_key"]]["signals"]]
        report = naive.compare(runs, reference=runs[0].name)
        self.assertEqual(report["pairwise"][0]["box_agreement"], 1.0)
        keys = naive.worst_frames(report, limit=2, runs=runs)
        self.assertEqual(len(keys), 2, "fall back to a sample when nothing disagrees")
        self.assertEqual(naive.worst_frames(report, limit=2), [],
                         "without runs there is nothing to fall back to")

    def test_crop_uses_the_union_over_all_configs(self):
        out, _ = self.run_matrix()
        runs = naive.load_runs_from_manifest(out / "run_manifest.json")
        present = [(r.name, r.frames["00000"]) for r in runs]
        region = grid_mod.detection_crop(present, (100, 100, 3), min_size=10)
        x0, y0, x1, y1 = region
        # wide's extra box at (60,60)-(80,80) must be inside the crop
        self.assertLessEqual(x0, 60)
        self.assertGreaterEqual(x1, 80)
        self.assertLessEqual(y0, 60)
        self.assertGreaterEqual(y1, 80)
        for _name, rec in present:
            rec["signals"] = []
        self.assertIsNone(grid_mod.detection_crop(present, (100, 100, 3)),
                          "no detections, no crop")

    def test_labels_are_drawn_at_panel_scale(self):
        out, _ = self.run_matrix()
        runs = naive.load_runs_from_manifest(out / "run_manifest.json")
        grid = grid_mod.render_frame_grid(runs, "00000", width=400)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.shape[1], 400, "two 200px panels")
        # the box lives at 10..30 of a 100px frame -> 20..60 of a 200px panel;
        # sample a row through its middle, below the label text
        panel = grid[grid_mod.HEADER_H:, :200]
        row = panel[40]
        red = ((row[:, 2] > 200) & (row[:, 0] < 80)).nonzero()[0]
        self.assertTrue(len(red), "the box edges should be drawn in the panel")
        self.assertLess(abs(red.min() - 20), 6, "left edge scaled to the panel")
        self.assertLess(abs(red.max() - 60), 6, "right edge scaled to the panel")


class MissingFramesTest(HarnessTestCase):
    def test_uneven_frame_sets_are_reported(self):
        out, _ = self.run_matrix()
        (out / "labels" / "wide" / "00003.json").unlink()
        runs = naive.load_runs_from_manifest(out / "run_manifest.json")
        report = naive.compare(runs, reference="narrow")
        self.assertEqual(report["frames"]["common"], 3)
        self.assertIsNotNone(report["frames"]["warning"])
        self.assertEqual(report["pairwise"][0]["frames_only_in_reference"], 1)


if __name__ == "__main__":
    unittest.main()
