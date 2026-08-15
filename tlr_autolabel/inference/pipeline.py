"""L1 inference pipeline: one configuration, one frame in, Tier A payload out.

This is the orchestration that used to live inside `cli/autolabel.py:main` —
model construction, the per-frame detect->crop->classify pass, the provenance
`meta` block, and the `tlr_autolabel/v1` payload assembly. Pulling it out of
`main` is what lets a second caller (the comparison runner) hold several
configurations and feed them from a non-directory frame source, without
duplicating any of the payload contract.

Tier A stays `tlr_autolabel/v1`: the only additions are optional keys that are
absent unless asked for (`source` for non-file frame sources, `timing_ms` with
`record_timing`, model digests with `record_model_digest`), so existing
consumers and existing label directories are unaffected.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter

from tlr_autolabel.inference.config import (
    InferenceConfig, file_digest, model_root, validate_config,
)
from tlr_autolabel.inference.detector import (
    Detector, clipped_detection_box, detect_full_and_tiles,
)
from tlr_autolabel.inference.lamp_recognizer import (
    LampClassifier, normalize_lamps, signal_state,
)

SCHEMA_VERSION = "tlr_autolabel/v1"


def process_image_with_candidates(img, detector, classifier, args, stats=None):
    """Detect, then classify each kept box. `args` is an `InferenceConfig` (or
    anything with the same attribute names — the CLI Namespace used to be it).

    `stats`, when given, accumulates per-stage timings and crop counts; it is
    the only addition to the historical behavior and is inert when omitted.
    """
    ih, iw = img.shape[:2]
    score_thr = (args.det_low_score_thr
                 if args.det_low_score_thr is not None
                 else args.det_score_thr)
    t0 = perf_counter()
    boxes = detect_full_and_tiles(detector, img, args, score_thr=score_thr)
    if stats is not None:
        stats["detector_ms"] = stats.get("detector_ms", 0.0) + (perf_counter() - t0) * 1e3

    results = []
    raw_detections = []
    for b in boxes:
        box_xyxy = clipped_detection_box(b, iw, ih, args.min_box)
        if box_xyxy is None:
            continue
        is_high = b["prob"] >= args.det_score_thr
        lamps = []
        if classifier is not None and (is_high or args.classify_low_detections):
            t1 = perf_counter()
            lamps = normalize_lamps(classifier.classify(img, box_xyxy))
            if stats is not None:
                stats["classifier_ms"] = stats.get("classifier_ms", 0.0) \
                    + (perf_counter() - t1) * 1e3
                stats["crops"] = stats.get("crops", 0) + 1
        candidate = {
            "detector_score": round(b["prob"], 4),
            "box_xyxy": list(box_xyxy),
            "lamps": lamps,
            "state": signal_state(lamps),
            "detection_level": "high" if is_high else "low",
        }
        raw_detections.append(dict(candidate))
        if is_high:
            if args.drop_unknown and not lamps:
                continue
            results.append({
                "detector_score": candidate["detector_score"],
                "box_xyxy": candidate["box_xyxy"],
                "lamps": candidate["lamps"],
                "state": candidate["state"],
            })
    return results, raw_detections


def process_image(img, detector, classifier, args):
    return process_image_with_candidates(img, detector, classifier, args)[0]


def new_run_id() -> str:
    return (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-" + uuid.uuid4().hex[:8])


@dataclass
class Pipeline:
    """A built detector+classifier pair plus the payload/meta contract.

    Construct with `Pipeline.build(cfg)`; pass models in directly when they are
    already built (the CLI does, so its monkeypatch seam is unchanged).
    """

    cfg: InferenceConfig
    detector: object
    classifier: object | None
    run_id: str = ""

    def __post_init__(self):
        if not self.run_id:
            self.run_id = new_run_id()
        # the wrapper resolved the family (explicit or inferred); record it
        self.detector_kind = getattr(self.detector, "kind", None) or self.cfg.detector_type
        self._meta = None

    @classmethod
    def build(cls, cfg: InferenceConfig, run_id: str = "") -> "Pipeline":
        validate_config(cfg)
        detector = Detector(cfg.detector, cfg.comlops_param, model_type=cfg.detector_type)
        if detector.kind == "comlops":
            detector.set_keep_classes(
                [s.strip() for s in cfg.det_classes.split(",") if s.strip()])
        classifier = None
        if cfg.classifier_enabled:
            classifier = LampClassifier(cfg.classifier, cfg.classifier_param, cfg,
                                        model_type=cfg.classifier_type)
        return cls(cfg=cfg, detector=detector, classifier=classifier, run_id=run_id)

    def close(self):
        """Release model runtimes (TensorRT helper processes in particular) so a
        comparison run can load the next configuration without piling engines up."""
        for model in (self.detector, self.classifier):
            trt = getattr(model, "trt", None)
            if trt is not None and hasattr(trt, "close"):
                trt.close()

    # ------------------------------------------------------------- description

    @property
    def detector_backend(self) -> str:
        if getattr(self.detector, "sess", None) is None:
            return "tensorrt-engine"
        return str(self.detector.sess.get_providers())

    @property
    def classifier_backend(self) -> str | None:
        return getattr(self.classifier, "backend", None) if self.classifier else None

    def describe(self) -> str:
        cfg = self.cfg
        lines = [f"detector={os.path.basename(cfg.detector)} "
                 f"[{getattr(self.detector, 'kind', cfg.detector_type)}] "
                 f"in={self.detector.w}x{self.detector.h} backend={self.detector_backend}"]
        if self.classifier is not None:
            lines.append(f"classifier={os.path.basename(cfg.classifier)} "
                         f"in={self.classifier.width}x{self.classifier.height} "
                         f"backend={self.classifier_backend}")
        else:
            lines.append("classifier=none (detector-only run: state is always unknown)")
        return "\n".join(lines)

    # ------------------------------------------------------------------- meta

    def meta(self) -> dict:
        """The provenance block written into every per-image JSON. Key order is
        part of the historical output; new keys are appended, never inserted."""
        if self._meta is not None:
            return self._meta
        cfg = self.cfg
        meta = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "preset": cfg.preset,
            "detector": os.path.basename(cfg.detector),
            "detector_backend": self.detector_backend,
            "model_root": model_root(),
            "classifier": os.path.basename(cfg.classifier) if cfg.classifier else None,
            "classifier_backend": self.classifier_backend,
            "tiles": bool(cfg.tiles),
            "det_score_thr": cfg.det_score_thr,
            "det_low_score_thr": cfg.det_low_score_thr,
            "classify_low_detections": bool(cfg.classify_low_detections),
            "det_nms_thr": cfg.det_nms_thr,
            "cls_score_thr": cfg.cls_score_thr,
            "cls_nms_thr": cfg.cls_nms_thr,
            "min_box": cfg.min_box,
            "crop_pad": cfg.crop_pad,
        }
        if cfg.detector_type or getattr(self, "detector_kind", None):
            meta["detector_type"] = cfg.detector_type or self.detector_kind
        if cfg.classifier_enabled and getattr(self.classifier, "kind", None):
            meta["classifier_type"] = self.classifier.kind
        if cfg.record_model_digest:
            meta["detector_sha256"] = file_digest(cfg.detector)
            meta["classifier_sha256"] = file_digest(cfg.classifier)
        self._meta = meta
        return meta

    # ------------------------------------------------------------------- run

    def run(self, frame) -> dict:
        """One frame -> one Tier A payload (not written to disk)."""
        stats = {} if self.cfg.record_timing else None
        t0 = perf_counter()
        results, raw_detections = process_image_with_candidates(
            frame.image, self.detector, self.classifier, self.cfg, stats=stats)
        total_ms = (perf_counter() - t0) * 1e3
        w, h = frame.size
        for i, r in enumerate(results):
            results[i] = {"signal_id": f"{os.path.basename(frame.frame_id)}-{i:02d}", **r}
        if self.cfg.det_low_score_thr is not None:
            for i, r in enumerate(raw_detections):
                raw_detections[i] = {
                    "raw_detection_id": f"{os.path.basename(frame.frame_id)}-raw-{i:02d}",
                    **r}
        payload = {"schema_version": SCHEMA_VERSION,
                   "image": frame.rel_path,
                   "image_realpath": frame.realpath,
                   "sample_data_token": frame.sample_data_token,
                   "channel": frame.channel,
                   "frame_index": frame.frame_index,
                   "width": w, "height": h,
                   "meta": self.meta(),
                   "signals": results}
        if frame.source is not None:
            payload["source"] = frame.source
        if frame.timestamp_us is not None:
            payload["timestamp_us"] = frame.timestamp_us
        if self.cfg.det_low_score_thr is not None:
            payload["raw_detections"] = raw_detections
        if stats is not None:
            payload["timing_ms"] = {
                "detector": round(stats.get("detector_ms", 0.0), 3),
                "classifier": round(stats.get("classifier_ms", 0.0), 3),
                "total": round(total_ms, 3),
                "crops": stats.get("crops", 0),
            }
        return payload
