#!/usr/bin/env python3
"""Render an editable RE timeline review HTML.

The page is static by default: opened over file:// it needs no backend, and
reviewers export a traffic_signal_re_review/v1 JSON file for apply_re_review.py.
Browsers cannot write local files from a file:// page, so `--serve` runs a
localhost server that also accepts the review JSON back and writes it to the
dataset in place.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import html
import json
import os
import shutil
from collections import Counter, defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DRAFT_ENDPOINT = "/__save_draft"
COMMIT_ENDPOINT = "/__commit_review"
DIFF_ENDPOINT = "/__review_diff"

DEFAULT_CROP_CHANNELS = "auto"

# t4devkit allows any CAM_* channel name, so the usable set is discovered from
# the data rather than hard-coded (a TLR dataset typically ships
# CAM_TRAFFIC_LIGHT_NEAR/FAR, not CAM_FRONT*). Only channels that explicitly
# face backwards are dropped: a rear camera cannot see a signal the ego is
# driving toward.
REAR_CHANNEL_MARKERS = {"REAR", "BACK", "BACKWARD"}


def id_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def sorted_ids(values) -> list[str]:
    return sorted((str(v) for v in values), key=id_sort_key)


def signal_group_id(member_ways: list[str]) -> str:
    return "ways:" + ",".join(sorted_ids(member_ways))


def split_ids(value: str) -> set[str]:
    return {v.strip() for v in (value or "").split(",") if v.strip()}


def parse_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def rel_href(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path, base_dir)).as_posix()


def companion_links(root: Path, output_dir: Path, **paths: Path) -> dict[str, str]:
    """Relative hrefs from one generated page to its sibling views.

    The three views are generated independently, so a link may point at a page
    that does not exist yet; that is deliberate, since requiring a generation
    order would be worse than a link the reviewer can regenerate into.
    """
    return {
        name: rel_href(path if path.is_absolute() else root / path, output_dir)
        for name, path in paths.items()
    }


def parse_channels(value: str) -> set[str] | None:
    if value.strip().lower() in {"all", "*"}:
        return None
    return {v.strip() for v in value.split(",") if v.strip()}


def is_rear_channel(channel: str) -> bool:
    """True for channels that explicitly face backwards (CAM_BACK, CAM_REAR_LEFT...).
    Matches whole underscore-separated tokens so a name that merely contains the
    letters (e.g. CAM_BACKUP_FRONT) is not dropped by accident."""
    return bool(set(str(channel).upper().split("_")) & REAR_CHANNEL_MARKERS)


def auto_crop_channels(channels) -> set[str]:
    """Forward-ish CAM_* channels discovered from the data itself."""
    return {
        ch for ch in channels
        if ch and str(ch).upper().startswith("CAM_") and not is_rear_channel(ch)
    }


def resolve_crop_channels(value: str, available) -> set[str] | None:
    """Resolve --crop-channels into a concrete set (None = no filtering).

    'auto' (default) discovers CAM_* channels present in the sidecar and drops
    rear-facing ones; 'all'/'*' disables filtering; anything else is an explicit
    comma-separated list."""
    if value.strip().lower() == "auto":
        return auto_crop_channels(available)
    return parse_channels(value)


def segment_observations(observations: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for obs in sorted(observations, key=lambda o: o.get("timestamp") or 0):
        state = obs.get("state", "unknown") or "unknown"
        if segments and segments[-1]["state"] == state:
            cur = segments[-1]
            cur["end_sample_token"] = obs["sample_token"]
            cur["end_timestamp"] = obs.get("timestamp")
            cur["sample_tokens"].append(obs["sample_token"])
            cur["n_frames"] += 1
            cur["flags"] = sorted(set(cur["flags"] + obs.get("flags", [])))
            cur["confidence_sum"] += float(obs.get("confidence") or 0.0)
            continue
        segments.append(
            {
                "start_sample_token": obs["sample_token"],
                "end_sample_token": obs["sample_token"],
                "sample_tokens": [obs["sample_token"]],
                "start_timestamp": obs.get("timestamp"),
                "end_timestamp": obs.get("timestamp"),
                "state": state,
                "n_frames": 1,
                "flags": sorted(obs.get("flags", [])),
                "confidence_sum": float(obs.get("confidence") or 0.0),
                "head_states": obs.get("head_states", {}),
            }
        )
    for segment in segments:
        segment["confidence"] = round(
            segment.pop("confidence_sum") / max(segment["n_frames"], 1), 3
        )
    return segments


def segment_visibility(observations: list[dict]) -> list[dict]:
    """Contiguous same-visibility runs for one (signal group, camera channel).

    Unlike segment_observations (fused cross-camera state), visibility is
    inherently per-camera -- a passing vehicle can occlude one camera's view
    of a signal while another camera sees it fine -- so this segments a
    single channel's own observation stream.
    """
    segments: list[dict] = []
    for obs in sorted(observations, key=lambda o: o["timestamp"]):
        visibility = obs["visibility"]
        if segments and segments[-1]["visibility"] == visibility:
            cur = segments[-1]
            cur["end_sample_token"] = obs["sample_token"]
            cur["end_timestamp"] = obs["timestamp"]
            cur["sample_tokens"].append(obs["sample_token"])
            cur["n_frames"] += 1
            continue
        segments.append(
            {
                "start_sample_token": obs["sample_token"],
                "end_sample_token": obs["sample_token"],
                "sample_tokens": [obs["sample_token"]],
                "start_timestamp": obs["timestamp"],
                "end_timestamp": obs["timestamp"],
                "visibility": visibility,
                "n_frames": 1,
            }
        )
    return segments


def visibility_observations_by_channel(
    annotations: list[dict], row: dict
) -> dict[str, list[dict]]:
    by_channel: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations:
        if not annotation_matches_group(ann, row):
            continue
        timestamp = ann.get("timestamp")
        if timestamp is None:
            continue
        attrs = ann.get("attributes") or {}
        by_channel[ann.get("channel", "")].append(
            {
                "timestamp": timestamp,
                "sample_token": ann.get("sample_token", ""),
                "visibility": attrs.get("visibility") or "unknown",
            }
        )
    return by_channel


def build_visibility_tracks(
    annotations: list[dict], rows: list[dict], crop_channels: set[str] | None
) -> None:
    """Attach row["visibility_tracks"] = {channel: [segments]} in place.

    Restricted to the same channel set as the crop candidates (crop_channels)
    so the timeline doesn't grow a track per wide-angle/irrelevant camera.
    """
    for row in rows:
        by_channel = visibility_observations_by_channel(annotations, row)
        tracks = {}
        for channel, observations in sorted(by_channel.items()):
            if crop_channels is not None and channel not in crop_channels:
                continue
            tracks[channel] = segment_visibility(observations)
        row["visibility_tracks"] = tracks


def build_rows(series: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for item in series:
        grouped[tuple(sorted_ids(item["member_ways"]))].append(item)

    rows = []
    for member_ways, members in sorted(grouped.items(), key=lambda kv: kv[0]):
        representative = max(members, key=lambda s: s.get("n_observations", 0))
        rel_ids = sorted_ids(s["regulatory_element_id"] for s in members)
        segments = segment_observations(representative.get("observations", []))
        rows.append(
            {
                "signal_group_id": signal_group_id(list(member_ways)),
                "member_ways": list(member_ways),
                "regulatory_element_ids": rel_ids,
                "label": "ways " + ",".join(member_ways),
                "sublabel": "RE " + ",".join(rel_ids),
                "segments": segments,
            }
        )
    return rows


def annotation_matches_group(ann: dict, row: dict) -> bool:
    attrs = ann.get("attributes") or {}
    way_id = attrs.get("map_traffic_light_id", "")
    if way_id and way_id in set(row["member_ways"]):
        return True
    ann_re_ids = split_ids(attrs.get("regulatory_element_id", ""))
    return bool(ann_re_ids & set(row["regulatory_element_ids"]))


def annotation_rank(ann: dict, segment: dict) -> float:
    attrs = ann.get("attributes") or {}
    box = ann.get("box2d") or ann.get("bbox") or [0, 0, 0, 0]
    w = max(float(box[2]) - float(box[0]), 0.0)
    h = max(float(box[3]) - float(box[1]), 0.0)
    min_side_score = min(min(w, h) / 80.0, 1.0) if w and h else 0.0
    detector_score = parse_float(attrs.get("detector_score"), 0.0)
    visibility = attrs.get("visibility", "")
    visibility_score = {
        "clear": 1.0,
        "full": 1.0,
        "partial_occluded": 0.55,
        "heavy_occluded": 0.2,
    }.get(visibility, 0.45)
    source_weight = {
        "auto": 1.0,
        "tracked": 0.9,
        "manual": 0.95,
        "cvat": 0.95,
        "projected_map": 0.75,
        "propagated": 0.55,
        "interpolated": 0.6,
        "map_presence": 0.45,
    }.get(attrs.get("source_type", ""), 0.75)
    sample_tokens = segment.get("sample_tokens") or []
    if ann.get("sample_token") in sample_tokens:
        if len(sample_tokens) == 1:
            centrality = 1.0
        else:
            index = sample_tokens.index(ann["sample_token"])
            center_index = (len(sample_tokens) - 1) / 2
            centrality = 1.0 - min(abs(index - center_index) / max(center_index, 1.0), 1.0)
    else:
        start_ts = int(segment["start_timestamp"])
        end_ts = int(segment["end_timestamp"])
        duration = max(end_ts - start_ts, 1)
        center = (start_ts + end_ts) / 2
        centrality = 1.0 - min(abs(int(ann.get("timestamp") or center) - center) / duration, 1.0)
    return round(
        source_weight
        * (
            0.45 * min_side_score
            + 0.35 * detector_score
            + 0.10 * visibility_score
            + 0.10 * centrality
        ),
        4,
    )


def annotations_in_segment(annotations: list[dict], segment: dict) -> list[dict]:
    sample_tokens = set(segment.get("sample_tokens") or [])
    def in_segment(ann: dict) -> bool:
        if sample_tokens:
            return ann.get("sample_token") in sample_tokens
        return segment["start_timestamp"] <= int(ann.get("timestamp") or 0) <= segment["end_timestamp"]
    return [ann for ann in annotations if in_segment(ann)]


def select_crop_candidates(annotations: list[dict], segment: dict, limit: int) -> list[dict]:
    ranked = sorted(
        (
            (annotation_rank(ann, segment), ann)
            for ann in annotations_in_segment(annotations, segment)
        ),
        key=lambda item: (-item[0], item[1].get("channel", ""), item[1].get("timestamp", 0)),
    )
    selected: list[tuple[float, dict]] = []
    used_channels: set[str] = set()
    seen_tokens: set[str] = set()
    for score, ann in ranked:
        token = ann.get("token", "")
        if token in seen_tokens:
            continue
        channel = ann.get("channel", "")
        if channel in used_channels:
            continue
        selected.append((score, ann))
        seen_tokens.add(token)
        used_channels.add(channel)
        if len(selected) >= limit:
            return selected
    for score, ann in ranked:
        token = ann.get("token", "")
        if token in seen_tokens:
            continue
        selected.append((score, ann))
        seen_tokens.add(token)
        if len(selected) >= limit:
            break
    return selected


def write_crop(root: Path, ann: dict, crop_path: Path, margin_ratio: float) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    image_path = root / ann["filename"]
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in ann["box2d"]]
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    margin = max(bw, bh) * margin_ratio
    cx0 = max(int(x0 - margin), 0)
    cy0 = max(int(y0 - margin), 0)
    cx1 = min(int(x1 + margin), width)
    cy1 = min(int(y1 + margin), height)
    if cx1 <= cx0 or cy1 <= cy0:
        return False
    crop = image[cy0:cy1, cx0:cx1].copy()
    scale = min(max(220.0 / max(min(crop.shape[:2]), 1), 1.0), 5.0)
    if scale > 1.01:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    rx0 = int(round((x0 - cx0) * scale))
    ry0 = int(round((y0 - cy0) * scale))
    rx1 = int(round((x1 - cx0) * scale))
    ry1 = int(round((y1 - cy0) * scale))
    thickness = max(2, int(round(scale * 1.5)))
    cv2.rectangle(crop, (rx0, ry0), (rx1, ry1), (0, 255, 255), thickness)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(crop_path), crop))


def build_roi_frames(
    annotations: list[dict], segment: dict, output_dir: Path, root: Path
) -> dict[str, list[dict]]:
    """Every annotated frame in `segment`, keyed by channel, for ROI correction.

    Unlike select_crop_candidates (a ranked top-N subset for quick evidence),
    ROI editing needs every frame so a reviewer can step through and fix a
    box that only drifts wrong for a few frames of an otherwise-fine run.
    """
    by_channel: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations_in_segment(annotations, segment):
        if not ann.get("box2d") or not ann.get("filename"):
            continue
        by_channel[ann.get("channel", "")].append(
            {
                "token": ann.get("token", ""),
                "channel": ann.get("channel", ""),
                "sample_token": ann.get("sample_token", ""),
                "timestamp": ann.get("timestamp"),
                "box2d": [round(float(v), 1) for v in ann["box2d"]],
                "full_image": rel_href(root / ann["filename"], output_dir),
                "filename": ann.get("filename", ""),
            }
        )
    for frames in by_channel.values():
        frames.sort(key=lambda f: f.get("timestamp") or 0)
    return dict(by_channel)


def attach_crop_candidates(
    root: Path,
    rows: list[dict],
    annotations: list[dict],
    assets_dir: Path,
    output_dir: Path,
    limit: int,
    margin_ratio: float,
    crop_channels: set[str] | None,
) -> int:
    if limit <= 0:
        return 0
    total = 0
    for row in rows:
        group_annotations = [
            ann for ann in annotations
            if annotation_matches_group(ann, row) and ann.get("box2d") and ann.get("filename")
            and (crop_channels is None or ann.get("channel") in crop_channels)
        ]
        for segment_index, segment in enumerate(row["segments"]):
            candidates = []
            for rank, ann in select_crop_candidates(group_annotations, segment, limit):
                key = hashlib.sha1(
                    f"{row['signal_group_id']}:{segment_index}:{ann.get('token')}".encode()
                ).hexdigest()[:16]
                crop_path = assets_dir / f"{key}.jpg"
                if not crop_path.exists() and not write_crop(root, ann, crop_path, margin_ratio):
                    continue
                box = [round(float(v), 1) for v in ann["box2d"]]
                attrs = ann.get("attributes") or {}
                area = round(max(box[2] - box[0], 0) * max(box[3] - box[1], 0), 1)
                candidates.append(
                    {
                        "src": rel_href(crop_path, output_dir),
                        "full_image": rel_href(root / ann["filename"], output_dir),
                        "token": ann.get("token", ""),
                        "channel": ann.get("channel", ""),
                        "filename": ann.get("filename", ""),
                        "sample_token": ann.get("sample_token", ""),
                        "timestamp": ann.get("timestamp"),
                        "state": attrs.get("state", ""),
                        "raw_state": attrs.get("raw_state", ""),
                        "source_type": attrs.get("source_type", ""),
                        "visibility": attrs.get("visibility", ""),
                        "detector_score": attrs.get("detector_score", ""),
                        "rank_score": rank,
                        "box_area": area,
                        "box2d": box,
                    }
                )
            segment["candidates"] = candidates
            segment["roi_frames"] = build_roi_frames(group_annotations, segment, output_dir, root)
            total += len(candidates)
    return total


def hide_segments_without_crops(rows: list[dict]) -> tuple[list[dict], int]:
    filtered_rows = []
    hidden_segments = 0
    for row in rows:
        kept_segments = [segment for segment in row["segments"] if segment.get("candidates")]
        hidden_segments += len(row["segments"]) - len(kept_segments)
        if kept_segments:
            filtered_rows.append({**row, "segments": kept_segments})
    return filtered_rows, hidden_segments


def load_optional_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def validate_review_payload(payload) -> str | None:
    """Reject anything that is not a review document, so a stray POST cannot
    truncate the reviewer's file. Returns an error message, or None if OK."""
    if not isinstance(payload, dict):
        return "payload is not a JSON object"
    version = payload.get("schema_version")
    if version != "traffic_signal_re_review/v1":
        return f"unexpected schema_version: {version!r}"
    if not isinstance(payload.get("groups"), list):
        return "payload has no 'groups' list"
    return None


def write_review_file(review_out: Path, payload: dict, backup: bool = True) -> None:
    """Write atomically, optionally keeping one generation of backup."""
    review_out.parent.mkdir(parents=True, exist_ok=True)
    if backup and review_out.exists():
        shutil.copy2(review_out, review_out.with_suffix(review_out.suffix + ".bak"))
    tmp = review_out.with_suffix(review_out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, review_out)


def draft_path_for(review_out: Path) -> Path:
    """Work-in-progress file that auto-save writes, next to the real one."""
    return review_out.with_name(f"{review_out.stem}.draft{review_out.suffix}")


# Fields that decide whether two decisions for the same interval differ.
# Bookkeeping like `source` and `n_frames` is derived, not reviewer intent.
COMPARED_FIELDS = {
    "state": ("state", "review_status", "note"),
    "visibility": ("visibility", "review_status", "note"),
    "roi": ("box2d", "review_status", "note"),
}


def index_review(payload: dict) -> dict[str, dict]:
    """Key every decision so two review documents can be compared entry by
    entry rather than as opaque blobs."""
    state: dict = {}
    visibility: dict = {}
    roi: dict = {}
    for group in payload.get("groups") or []:
        gid = group.get("signal_group_id")
        for d in group.get("decisions") or []:
            key = (gid, d.get("start_sample_token"), d.get("end_sample_token"))
            state[key] = d
        for channel, segs in (group.get("visibility_decisions") or {}).items():
            for d in segs or []:
                key = (gid, channel, d.get("start_sample_token"), d.get("end_sample_token"))
                visibility[key] = d
        for d in group.get("roi_decisions") or []:
            roi[(gid, d.get("annotation_token"))] = d
    return {"state": state, "visibility": visibility, "roi": roi}


def short_token(value) -> str:
    text = "" if value is None else str(value)
    return text[:8] if len(text) > 8 else text


def entry_label(kind: str, key: tuple) -> str:
    if kind == "state":
        gid, start, end = key
        return f"{gid}  {short_token(start)}..{short_token(end)}"
    if kind == "visibility":
        gid, channel, start, end = key
        return f"{gid} ({channel})  {short_token(start)}..{short_token(end)}"
    gid, token = key
    return f"{gid}  ann {short_token(token)}"


def entry_fields(kind: str, decision: dict) -> dict:
    return {field: decision.get(field) for field in COMPARED_FIELDS[kind]}


def diff_reviews(old: dict, new: dict) -> dict:
    """What committing `new` would change in `old`, per decision kind."""
    old_index = index_review(old)
    new_index = index_review(new)
    result: dict = {}
    for kind in ("state", "visibility", "roi"):
        before, after = old_index[kind], new_index[kind]
        added, removed, changed = [], [], []
        for key in after:
            if key not in before:
                added.append({"label": entry_label(kind, key),
                              "after": entry_fields(kind, after[key])})
        for key in before:
            if key not in after:
                removed.append({"label": entry_label(kind, key),
                                "before": entry_fields(kind, before[key])})
        for key in after:
            if key not in before:
                continue
            old_fields = entry_fields(kind, before[key])
            new_fields = entry_fields(kind, after[key])
            if old_fields != new_fields:
                changed.append({"label": entry_label(kind, key),
                                "before": old_fields, "after": new_fields})
        result[kind] = {"added": added, "removed": removed, "changed": changed}
    result["total"] = sum(
        len(result[kind][bucket])
        for kind in ("state", "visibility", "roi")
        for bucket in ("added", "removed", "changed")
    )
    return result


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    """Static file server for the dataset root, plus the review endpoints.

    Auto-save only ever touches the draft file. The real review file changes
    exactly once per explicit commit, so an interrupted session leaves the
    reviewed output untouched and resumable from the draft.
    """

    review_out: Path = Path()
    draft_out: Path = Path()
    root: Path = Path()

    def _send_json(self, status: int, body: dict) -> None:
        blob = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _read_payload(self):
        """Returns (payload, error_message)."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length))
        except (ValueError, TypeError) as exc:
            return None, f"invalid JSON: {exc}"
        return payload, validate_review_payload(payload)

    def _rel(self, path: Path) -> str:
        return os.path.relpath(path, self.root)

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        if self.path not in (DRAFT_ENDPOINT, COMMIT_ENDPOINT, DIFF_ENDPOINT):
            self._send_json(404, {"ok": False, "error": "unknown endpoint"})
            return
        payload, error = self._read_payload()
        if error:
            self._send_json(400, {"ok": False, "error": error})
            return

        if self.path == DIFF_ENDPOINT:
            committed = load_optional_json(self.review_out)
            self._send_json(200, {
                "ok": True,
                "path": self._rel(self.review_out),
                "committed_exists": self.review_out.exists(),
                "diff": diff_reviews(committed, payload),
            })
            return

        if self.path == DRAFT_ENDPOINT:
            try:
                # No .bak for the draft: it is rewritten constantly and the
                # committed file is the thing worth protecting.
                write_review_file(self.draft_out, payload, backup=False)
            except OSError as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, {"ok": True, "path": self._rel(self.draft_out)})
            return

        try:
            write_review_file(self.review_out, payload)
            self.draft_out.unlink(missing_ok=True)
        except OSError as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})
            return
        n_groups = len(payload.get("groups", []))
        print(f"committed {self.review_out} ({n_groups} groups)", flush=True)
        self._send_json(200, {"ok": True, "path": self._rel(self.review_out)})

    def log_message(self, fmt, *args):  # keep the console readable
        return


def serve_review(
    root: Path, output_path: Path, review_out: Path, draft_out: Path, port: int
) -> None:
    handler = functools.partial(ReviewRequestHandler, directory=str(root))
    ReviewRequestHandler.review_out = review_out
    ReviewRequestHandler.draft_out = draft_out
    ReviewRequestHandler.root = root
    rel_page = os.path.relpath(output_path, root)
    # Bind loopback only: these endpoints write to the dataset.
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {root} at http://127.0.0.1:{port}/", flush=True)
        print(f"open http://127.0.0.1:{port}/{rel_page}", flush=True)
        print(f"auto-save writes {draft_out}", flush=True)
        print(f"'Export / commit' writes {review_out}", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument(
        "--input",
        default=Path("annotation/traffic_signal_re_timeseries.json"),
        type=Path,
    )
    parser.add_argument(
        "--review",
        default=None,
        type=Path,
        help="optional existing traffic_signal_re_review/v1 JSON to pre-load",
    )
    parser.add_argument(
        "--sidecar",
        default=Path("annotation/traffic_signal_2d_ann.json"),
        type=Path,
        help="Tier B sidecar used to generate representative crop candidates",
    )
    parser.add_argument(
        "--assets-dir",
        default=Path("build/tl_match/re_review_assets"),
        type=Path,
        help="directory for generated crop images",
    )
    parser.add_argument(
        "--crop-candidates",
        default=6,
        type=int,
        help="maximum crop candidates embedded per timeline segment",
    )
    parser.add_argument(
        "--crop-margin",
        default=1.2,
        type=float,
        help="crop margin as a multiple of the bbox longer side",
    )
    parser.add_argument(
        "--crop-channels",
        default=DEFAULT_CROP_CHANNELS,
        help="camera channels used for crop candidates: 'auto' (default; every "
             "CAM_* channel found in the sidecar except rear-facing ones), 'all' "
             "to disable filtering, or an explicit comma-separated list",
    )
    parser.add_argument(
        "--show-empty-crop-segments",
        action="store_true",
        help="show timeline segments that have no crop candidates after channel filtering",
    )
    parser.add_argument(
        "--output",
        default=Path("build/tl_match/re_review_timeline.html"),
        type=Path,
    )
    parser.add_argument(
        "--frame-view",
        default=Path("build/tl_match/re_frame_view.html"),
        type=Path,
        help="companion frame view to link to (generated by re_frame_view.py)",
    )
    parser.add_argument(
        "--map-view",
        default=Path("build/tl_match/re_map_view.html"),
        type=Path,
        help="companion map view to link to (generated by re_map_view.py)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="serve the page over localhost so it can write the review JSON back to disk",
    )
    parser.add_argument("--port", default=8765, type=int, help="port used by --serve")
    parser.add_argument(
        "--review-out",
        default=Path("annotation/traffic_signal_re_review.json"),
        type=Path,
        help="path --serve writes when the page saves; existing file is backed up to .bak",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    input_path = args.input if args.input.is_absolute() else root / args.input
    review_path = None
    if args.review is not None:
        review_path = args.review if args.review.is_absolute() else root / args.review
    output_path = args.output if args.output.is_absolute() else root / args.output
    sidecar_path = args.sidecar if args.sidecar.is_absolute() else root / args.sidecar
    assets_dir = args.assets_dir if args.assets_dir.is_absolute() else root / args.assets_dir
    review_out = args.review_out if args.review_out.is_absolute() else root / args.review_out

    if args.serve and not output_path.is_relative_to(root):
        # Compared logically, not via resolve(): datasets commonly symlink
        # data/ and build/ out to a shared sibling, and the served paths
        # follow those links just fine.
        raise SystemExit(
            f"--serve needs --output inside --dataset-root ({root}); "
            f"got {output_path}"
        )
    draft_out = draft_path_for(review_out)
    # Resume source, unless --review names one explicitly: the draft holds
    # in-progress work, so it wins over the last committed file.
    resumed_from = None
    if args.serve and review_path is None:
        if draft_out.exists():
            review_path = draft_out
            resumed_from = "draft"
        elif review_out.exists():
            review_path = review_out
            resumed_from = "committed"

    timeseries = json.loads(input_path.read_text())
    rows = build_rows(timeseries.get("series", []))
    sidecar_annotations = (
        json.loads(sidecar_path.read_text()).get("annotations", [])
        if sidecar_path.exists() else []
    )
    crop_channels = resolve_crop_channels(
        args.crop_channels,
        {a.get("channel") for a in sidecar_annotations if a.get("channel")},
    )
    crop_channels_label = (
        "all" if crop_channels is None else (",".join(sorted(crop_channels)) or "none")
    )
    n_crops = attach_crop_candidates(
        root,
        rows,
        sidecar_annotations,
        assets_dir,
        output_path.parent,
        args.crop_candidates,
        args.crop_margin,
        crop_channels,
    )
    build_visibility_tracks(sidecar_annotations, rows, crop_channels)
    hidden_segments = 0
    if not args.show_empty_crop_segments:
        rows, hidden_segments = hide_segments_without_crops(rows)

    t_values = [
        s["start_timestamp"]
        for row in rows
        for s in row["segments"]
        if s.get("start_timestamp") is not None
    ]
    t_values += [
        s["end_timestamp"]
        for row in rows
        for s in row["segments"]
        if s.get("end_timestamp") is not None
    ]
    if not t_values:
        # Nearly always a channel-name mismatch, so name the channels the
        # sidecar actually has instead of leaving the reviewer to guess.
        available = sorted({
            ann.get("channel") for ann in sidecar_annotations if ann.get("channel")
        })
        raise SystemExit(
            "no timeline segments with crop candidates.\n"
            f"  --crop-channels {args.crop_channels} resolved to: {crop_channels_label}\n"
            f"  channels in {sidecar_path.name}: {', '.join(available) or 'none'}\n"
            "Pass --crop-channels with one of the channels above (or 'all'), "
            "or --show-empty-crop-segments to keep no-evidence segments."
        )
    flag_counts = Counter(
        flag for row in rows for seg in row["segments"] for flag in seg.get("flags", [])
    )

    data_json = json.dumps(
        {
            "rows": rows,
            "t0": min(t_values),
            "t1": max(t_values),
            "source_timeseries": str(args.input),
            "crop_channels": crop_channels_label,
            "hidden_empty_crop_segments": hidden_segments,
        },
        ensure_ascii=False,
    )
    review_json = json.dumps(load_optional_json(review_path), ensure_ascii=False)
    flag_summary = ", ".join(f"{k}: {v}" for k, v in flag_counts.most_common())
    subtitle = (
        f"{len(rows)} signal groups &middot; "
        f"{sum(len(r['segments']) for r in rows)} segments &middot; "
        f"{n_crops} crop candidates &middot; "
        f"crop channels: {html.escape(crop_channels_label)} &middot; "
        + (f"hidden no-evidence segments: {hidden_segments} &middot; " if hidden_segments else "")
        +
        f"flags: {html.escape(flag_summary or 'none')}"
    )

    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TLR RE timeline review</title>
<style>
  :root { --bg:#f8f8f6; --panel:#ffffff; --ink:#202428; --muted:#626a73;
    --line:#d8dde2; --red:#c7352b; --amber:#d49716; --green:#27824a;
    --unknown:#9aa3ad; --accent:#2457c5; --bad:#9f2d20;
    --vis-full:#8fb996; --vis-partial:#d49716; --vis-occluded:#c7352b; --vis-unknown:#c3c9cf;
    --aside-w:430px; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:13px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
  header { position:sticky; top:0; z-index:5; padding:12px 16px;
    background:var(--panel); border-bottom:1px solid var(--line); }
  h1 { margin:0 0 2px; font-size:17px; font-weight:650; }
  .sub { color:var(--muted); font-size:12px; }
  main { display:grid; grid-template-columns:minmax(0,1fr) 5px var(--aside-w); gap:0; }
  .resizer { position:sticky; top:58px; height:calc(100vh - 58px);
    cursor:col-resize; background:var(--line); touch-action:none; }
  .resizer:hover, .resizer.dragging { background:var(--accent); }
  #chart { overflow:auto; padding:16px; }
  .row { display:flex; align-items:center; min-width:max-content; margin-bottom:8px; }
  .rowlabel { flex:0 0 230px; padding-right:12px; font-size:12px; }
  .rowlabel b { display:block; font-weight:650; }
  .rowlabel span { color:var(--muted); font-size:11px; }
  .strip { position:relative; height:38px; border-left:1px solid var(--line); }
  .seg { position:absolute; top:7px; height:24px; border-radius:3px;
    box-sizing:border-box; border:1px solid rgba(0,0,0,.16); cursor:pointer;
    overflow:hidden; white-space:nowrap; text-overflow:clip; color:#fff;
    padding:3px 5px; font-size:11px; line-height:17px; }
  .seg:hover { outline:2px solid var(--ink); z-index:3; }
  .seg.selected { outline:3px solid var(--accent); z-index:4; }
  .seg.decided { box-shadow:inset 0 -4px 0 rgba(255,255,255,.75);
    border:1px dashed rgba(0,0,0,.75); font-weight:700; }
  .seg.flagged::before { content:""; position:absolute; left:0; right:0; top:0;
    height:3px; background:#111; opacity:.8; }
  .visrow { display:flex; align-items:center; min-width:max-content; margin-bottom:3px; }
  .visrow .rowlabel { font-size:10px; color:var(--muted); padding-left:14px; }
  .visstrip { position:relative; height:16px; border-left:1px solid var(--line); }
  .seg.vis { top:2px; height:12px; padding:0; font-size:0; }
  aside { height:calc(100vh - 58px); overflow:auto; position:sticky; top:58px;
    border-left:1px solid var(--line); background:var(--panel); padding:14px; }
  label { display:block; margin:10px 0 4px; color:var(--muted); font-size:12px; }
  input, select, textarea, button { font:inherit; box-sizing:border-box; }
  input, select, textarea { width:100%; border:1px solid var(--line);
    border-radius:4px; padding:7px; background:#fff; color:var(--ink); }
  textarea { min-height:72px; resize:vertical; }
  button { border:1px solid var(--line); border-radius:4px; background:#fff;
    color:var(--ink); padding:7px 9px; cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.danger { color:var(--bad); }
  .buttons { display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; }
  .hint, .mono { color:var(--muted); font-size:12px; }
  .mono { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; word-break:break-all; }
  .cropPanel { border:1px solid var(--line); border-radius:4px; background:#fafafa;
    min-height:120px; padding:8px; }
  .cropPanel img { display:block; width:100%; max-height:340px; object-fit:contain;
    background:#111; border-radius:3px; }
  .candidateTabs { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }
  .candidateTabs button { padding:4px 7px; font-size:12px; }
  .candidateTabs button.active { border-color:var(--accent); color:var(--accent); }
  .candidateMeta { margin-top:7px; color:var(--muted); font-size:12px; }
  .viewlinks { display:flex; align-items:center; gap:8px; margin-top:8px; }
  .viewlink { color:var(--accent); text-decoration:none; font-size:12px;
    border:1px solid var(--line); border-radius:4px; padding:4px 8px; }
  .viewlink:hover { border-color:var(--accent); }
  .viewlink.small { padding:3px 7px; font-size:11px; }
  #decisionList { margin-top:12px; border-top:1px solid var(--line); padding-top:10px; }
  .decision { border:1px solid var(--line); border-radius:4px; padding:7px;
    margin-bottom:7px; background:#fafafa; }
  .decision b { display:block; font-size:12px; }
  .overlay { position:fixed; inset:0; background:rgba(20,24,28,.55); z-index:20;
    display:flex; align-items:center; justify-content:center; padding:24px; }
  .modal { background:var(--panel); border:1px solid var(--line); border-radius:6px;
    padding:18px; width:min(760px, 100%); max-height:86vh; overflow:auto; }
  .modal h2 { margin:0 0 4px; font-size:16px; }
  .diffBody { margin-top:12px; }
  .diffKind { margin-bottom:14px; }
  .diffKind h3 { margin:0 0 6px; font-size:13px; text-transform:uppercase;
    letter-spacing:.04em; color:var(--muted); }
  .diffEntry { border:1px solid var(--line); border-left-width:4px; border-radius:4px;
    padding:7px 9px; margin-bottom:6px; background:#fafafa; font-size:12px; }
  .diffEntry.added { border-left-color:var(--green); }
  .diffEntry.removed { border-left-color:var(--bad); }
  .diffEntry.changed { border-left-color:var(--amber); }
  .diffEntry .tag { font-weight:700; text-transform:uppercase; font-size:10px;
    letter-spacing:.05em; margin-right:6px; }
  .diffEntry .before { color:var(--bad); text-decoration:line-through; }
  .diffEntry .after { color:var(--green); }
  .roiEditor { border:1px solid var(--line); border-radius:4px; background:#fafafa;
    padding:8px; margin-top:8px; }
  .roiEditor canvas { display:block; width:100%; aspect-ratio:1/1; background:#111;
    border-radius:3px; touch-action:none; }
  .roiNav { display:flex; align-items:center; gap:8px; margin-top:6px; }
  .roiNav .hint { flex:1; text-align:center; }
  .roiCoords { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:6px; margin-top:8px; }
  .roiCoords input { padding:5px; font-size:12px; }
  .roiCoords label { margin:0 0 2px; font-size:10px; }
</style></head><body>
<header>
  <h1>Traffic signal RE timeline review</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="viewlinks">
    <a id="frameLink" class="viewlink" href="#">Frame view &rarr;</a>
    <a id="mapLink" class="viewlink" href="#">Map view &rarr;</a>
    <span class="hint">follow the selected evidence frame</span>
  </div>
</header>
<main>
  <section id="chart"></section>
  <div id="resizer" class="resizer" title="Drag to resize; double-click to reset"></div>
  <aside>
    <div class="hint">Click a state segment or a per-camera visibility segment, edit it, then add it to the review JSON.</div>
    <label>Signal group</label>
    <div id="selectedGroup" class="mono">none</div>
    <label>Interval</label>
    <div id="selectedInterval" class="mono">none</div>
    <label>Image evidence</label>
    <div id="cropPanel" class="cropPanel">
      <div class="hint">Select a segment to show representative crops.</div>
    </div>
    <div id="roiEditor" class="roiEditor" style="display:none">
      <canvas id="roiCanvas" width="400" height="400"></canvas>
      <div class="roiNav">
        <button type="button" id="roiPrev">&larr; Prev frame</button>
        <span id="roiFrameInfo" class="hint">-</span>
        <button type="button" id="roiNext">Next frame &rarr;</button>
      </div>
      <div class="roiCoords">
        <div><label for="roiX0">x0</label><input id="roiX0" type="number" step="0.1"></div>
        <div><label for="roiY0">y0</label><input id="roiY0" type="number" step="0.1"></div>
        <div><label for="roiX1">x1</label><input id="roiX1" type="number" step="0.1"></div>
        <div><label for="roiY1">y1</label><input id="roiY1" type="number" step="0.1"></div>
      </div>
      <label for="roiStatusInput">ROI review status</label>
      <select id="roiStatusInput">
        <option value="fixed">fixed</option>
        <option value="accepted">accepted</option>
        <option value="rejected">rejected</option>
        <option value="unchecked">unchecked</option>
      </select>
      <label for="roiNoteInput">ROI note</label>
      <textarea id="roiNoteInput"></textarea>
      <div class="buttons">
        <button class="primary" id="roiSave">Save ROI</button>
        <button id="roiReset">Reset to detected box</button>
      </div>
      <div class="buttons">
        <button id="roiCopyPrev">Copy ROI from prev frame</button>
        <button id="roiCopyNext">Copy ROI from next frame</button>
      </div>
      <div class="buttons">
        <button id="roiRecenter">Recenter zoom</button>
        <button class="danger" id="roiClear">Clear saved fix</button>
        <button id="roiClose">Close</button>
      </div>
    </div>
    <div id="stateField">
      <label for="stateInput">State</label>
      <input id="stateInput" placeholder="red-circle">
    </div>
    <div id="visibilityField" style="display:none">
      <label for="visibilityInput">Visibility</label>
      <select id="visibilityInput">
        <option value="full">full</option>
        <option value="partial">partial</option>
        <option value="occluded">occluded</option>
        <option value="unknown">unknown</option>
      </select>
    </div>
    <label for="statusInput">Review status</label>
    <select id="statusInput">
      <option value="accepted">accepted</option>
      <option value="fixed">fixed</option>
      <option value="rejected">rejected</option>
      <option value="unchecked">unchecked</option>
    </select>
    <label for="noteInput">Note</label>
    <textarea id="noteInput"></textarea>
    <div class="buttons">
      <button class="primary" id="addDecision">Add/update</button>
      <button class="danger" id="clearDecision">Clear</button>
    </div>
    <div class="buttons">
      <button class="primary" id="commitReview" style="display:none">Export / commit</button>
      <button id="exportReview">Export JSON</button>
      <label style="margin:0"><input id="importReview" type="file" accept="application/json" style="display:none">
        <button id="importButton" type="button">Import JSON</button></label>
    </div>
    <label id="autoSaveRow" style="display:none; font-weight:400; margin:6px 0 0">
      <input type="checkbox" id="autoSave" checked> auto-save draft on every change
    </label>
    <div id="saveStatus" class="hint"></div>
    <div id="decisionList"></div>
  </aside>
</main>
<div id="diffOverlay" class="overlay" style="display:none">
  <div class="modal">
    <h2 id="diffTitle">Review changes</h2>
    <div id="diffTarget" class="hint"></div>
    <div id="diffBody" class="diffBody"></div>
    <div class="buttons">
      <button class="primary" id="diffConfirm">Commit to file</button>
      <button id="diffCancel">Cancel</button>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const INITIAL_REVIEW = __REVIEW__;
// Non-null only when the page is served by `--serve`; a file:// page cannot
// write back to disk, so it falls back to the Export JSON download.
const SAVE_CONFIG = __SAVE_CONFIG__;
const LINKS = __LINKS__;
const PX_PER_S = 42;
const MIN_SEG_W = 5;
let selected = null;
let decisions = new Map();
let visDecisions = new Map();
let roiDecisions = new Map();
let evidenceIndex = 0;
let roiState = null;
const ROI_CURSORS = {
  nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize',
  n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize', move: 'move',
};

function keyFor(groupId, seg) {
  return `${groupId}|${seg.start_sample_token}|${seg.end_sample_token}`;
}

function visKeyFor(groupId, channel, seg) {
  return `${groupId}|${channel}|${seg.start_sample_token}|${seg.end_sample_token}`;
}

// Every frame this row has on `channel`, not just the ones inside the
// selected segment: a box that drifted at a segment boundary is usually
// fixed by copying the neighbouring frame, which often sits in the
// adjacent segment.
function mergedRoiFrames(row, seg, channel) {
  const merged = new Map();
  for (const s of row.segments || []) {
    for (const f of (s.roi_frames || {})[channel] || []) merged.set(f.token, f);
  }
  return [...merged.values()].sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
}

// Pure geometry helpers (image pixel space <-> canvas pixel space). Kept
// free of DOM access so they can be unit tested in isolation.
function cropWindow(box, imgW, imgH, padFactor, minSize) {
  const bw = Math.max(box[2] - box[0], 1);
  const bh = Math.max(box[3] - box[1], 1);
  const cx = (box[0] + box[2]) / 2;
  const cy = (box[1] + box[3]) / 2;
  let size = Math.max(Math.max(bw, bh) * padFactor, minSize);
  size = Math.min(size, Math.max(imgW, imgH));
  let half = size / 2;
  let x0 = cx - half, y0 = cy - half, x1 = cx + half, y1 = cy + half;
  if (x0 < 0) { x1 -= x0; x0 = 0; }
  if (y0 < 0) { y1 -= y0; y0 = 0; }
  if (x1 > imgW) { x0 -= (x1 - imgW); x1 = imgW; }
  if (y1 > imgH) { y0 -= (y1 - imgH); y1 = imgH; }
  return [Math.max(x0, 0), Math.max(y0, 0), Math.min(x1, imgW), Math.min(y1, imgH)];
}

function imageToCanvas(pt, win, canvasW, canvasH) {
  const sx = canvasW / Math.max(win[2] - win[0], 1e-6);
  const sy = canvasH / Math.max(win[3] - win[1], 1e-6);
  return [(pt[0] - win[0]) * sx, (pt[1] - win[1]) * sy];
}

function canvasToImage(pt, win, canvasW, canvasH) {
  const sx = (win[2] - win[0]) / canvasW;
  const sy = (win[3] - win[1]) / canvasH;
  return [win[0] + pt[0] * sx, win[1] + pt[1] * sy];
}

function clampBox(box, imgW, imgH) {
  let [x0, y0, x1, y1] = box;
  x0 = Math.max(0, Math.min(x0, imgW - 1));
  y0 = Math.max(0, Math.min(y0, imgH - 1));
  x1 = Math.max(x0 + 1, Math.min(x1, imgW));
  y1 = Math.max(y0 + 1, Math.min(y1, imgH));
  return [x0, y0, x1, y1];
}

function roiHandles(cx0, cy0, cx1, cy1) {
  const mx = (cx0 + cx1) / 2, my = (cy0 + cy1) / 2;
  return [
    {name: 'nw', x: cx0, y: cy0}, {name: 'n', x: mx, y: cy0}, {name: 'ne', x: cx1, y: cy0},
    {name: 'w', x: cx0, y: my}, {name: 'e', x: cx1, y: my},
    {name: 'sw', x: cx0, y: cy1}, {name: 's', x: mx, y: cy1}, {name: 'se', x: cx1, y: cy1},
  ];
}

function roiHitTest(px, py, cx0, cy0, cx1, cy1) {
  for (const h of roiHandles(cx0, cy0, cx1, cy1)) {
    if (Math.abs(px - h.x) <= 8 && Math.abs(py - h.y) <= 8) return h.name;
  }
  if (px > cx0 && px < cx1 && py > cy0 && py < cy1) return 'move';
  return null;
}

function stateStyle(state) {
  const hasRed = state.includes('red-');
  const hasAmber = state.includes('amber-');
  const hasGreen = state.includes('green-');
  const colors = [];
  if (hasRed) colors.push('var(--red)');
  if (hasAmber) colors.push('var(--amber)');
  if (hasGreen) colors.push('var(--green)');
  if (!colors.length) return 'background:var(--unknown)';
  if (colors.length === 1) return `background:${colors[0]}`;
  const stops = colors.map((c, i) => {
    const a = Math.round((i / colors.length) * 100);
    const b = Math.round(((i + 1) / colors.length) * 100);
    return `${c} ${a}% ${b}%`;
  }).join(',');
  return `background:linear-gradient(90deg,${stops})`;
}

function visStyle(visibility) {
  const color = {
    full: 'var(--vis-full)',
    partial: 'var(--vis-partial)',
    occluded: 'var(--vis-occluded)',
  }[visibility] || 'var(--vis-unknown)';
  return `background:${color}`;
}

function loadReview(payload) {
  decisions = new Map();
  visDecisions = new Map();
  roiDecisions = new Map();
  for (const group of payload.groups || []) {
    const gid = group.signal_group_id;
    for (const d of group.decisions || group.segments || []) {
      const k = `${gid}|${d.start_sample_token}|${d.end_sample_token || d.start_sample_token}`;
      decisions.set(k, {...d, signal_group_id: gid});
    }
    for (const [channel, segs] of Object.entries(group.visibility_decisions || {})) {
      for (const d of segs) {
        const k = `${gid}|${channel}|${d.start_sample_token}|${d.end_sample_token || d.start_sample_token}`;
        visDecisions.set(k, {...d, signal_group_id: gid, channel});
      }
    }
    for (const d of group.roi_decisions || []) {
      roiDecisions.set(d.annotation_token, {...d, signal_group_id: gid});
    }
  }
}

function renderChart() {
  const chart = document.getElementById('chart');
  chart.innerHTML = '';
  const span = Math.max((DATA.t1 - DATA.t0) / 1e6, 1);
  const width = Math.ceil(span * PX_PER_S) + 80;
  for (const row of DATA.rows) {
    const rowEl = document.createElement('div');
    rowEl.className = 'row';
    rowEl.innerHTML = `<div class="rowlabel"><b>${row.label}</b><span>${row.sublabel}</span></div>`;
    const strip = document.createElement('div');
    strip.className = 'strip';
    strip.style.width = width + 'px';
    for (const seg of row.segments) {
      const left = ((seg.start_timestamp - DATA.t0) / 1e6) * PX_PER_S;
      const dur = Math.max((seg.end_timestamp - seg.start_timestamp) / 1e6, 0.1);
      const segEl = document.createElement('div');
      segEl.className = 'seg';
      if (seg.flags && seg.flags.length) segEl.classList.add('flagged');
      // A staged decision overrides what the bar shows, so the chart reflects
      // the reviewed state rather than the detector's original guess.
      const decision = decisions.get(keyFor(row.signal_group_id, seg));
      if (decision) segEl.classList.add('decided');
      // renderChart() rebuilds every bar, so re-apply the selection highlight
      // that the old element carried.
      if (selected && selected.kind === 'state' && selected.row === row && selected.seg === seg) {
        segEl.classList.add('selected');
      }
      const shownState = decision ? decision.state : seg.state;
      segEl.style.left = left + 'px';
      segEl.style.width = Math.max(dur * PX_PER_S, MIN_SEG_W) + 'px';
      segEl.setAttribute('style', segEl.getAttribute('style') + ';' + stateStyle(shownState));
      const stateLine = decision && decision.state !== seg.state
        ? `${shownState} (was ${seg.state})` : shownState;
      segEl.title = `${stateLine}\\nframes=${seg.n_frames} conf=${seg.confidence}\\n${(seg.flags || []).join(', ')}`;
      if (parseFloat(segEl.style.width) > 70) segEl.textContent = shownState;
      segEl.addEventListener('click', () => selectSegment(row, seg, segEl));
      strip.appendChild(segEl);
    }
    rowEl.appendChild(strip);
    chart.appendChild(rowEl);

    for (const [channel, segs] of Object.entries(row.visibility_tracks || {})) {
      const visRowEl = document.createElement('div');
      visRowEl.className = 'visrow';
      visRowEl.innerHTML = `<div class="rowlabel">${channel} visibility</div>`;
      const visStrip = document.createElement('div');
      visStrip.className = 'visstrip';
      visStrip.style.width = width + 'px';
      for (const seg of segs) {
        const left = ((seg.start_timestamp - DATA.t0) / 1e6) * PX_PER_S;
        const dur = Math.max((seg.end_timestamp - seg.start_timestamp) / 1e6, 0.1);
        const segEl = document.createElement('div');
        segEl.className = 'seg vis';
        const visDecision = visDecisions.get(visKeyFor(row.signal_group_id, channel, seg));
        if (visDecision) segEl.classList.add('decided');
        if (selected && selected.kind === 'visibility' && selected.row === row
            && selected.channel === channel && selected.seg === seg) {
          segEl.classList.add('selected');
        }
        const shownVis = visDecision ? visDecision.visibility : seg.visibility;
        segEl.style.left = left + 'px';
        segEl.style.width = Math.max(dur * PX_PER_S, MIN_SEG_W) + 'px';
        segEl.setAttribute('style', segEl.getAttribute('style') + ';' + visStyle(shownVis));
        const visLine = visDecision && visDecision.visibility !== seg.visibility
          ? `${shownVis} (was ${seg.visibility})` : shownVis;
        segEl.title = `${channel}: ${visLine}\\nframes=${seg.n_frames}`;
        segEl.addEventListener('click', () => selectVisSegment(row, channel, seg, segEl));
        visStrip.appendChild(segEl);
      }
      visRowEl.appendChild(visStrip);
      chart.appendChild(visRowEl);
    }
  }
}

function selectSegment(row, seg, el) {
  closeRoiEditor();
  document.querySelectorAll('.seg.selected').forEach(n => n.classList.remove('selected'));
  el.classList.add('selected');
  selected = {kind: 'state', row, seg};
  const existing = decisions.get(keyFor(row.signal_group_id, seg));
  document.getElementById('stateField').style.display = '';
  document.getElementById('visibilityField').style.display = 'none';
  document.getElementById('selectedGroup').textContent = row.signal_group_id;
  document.getElementById('selectedInterval').textContent =
    `${seg.start_sample_token} .. ${seg.end_sample_token}`;
  document.getElementById('stateInput').value = existing?.state || seg.state;
  document.getElementById('statusInput').value = existing?.review_status || 'accepted';
  document.getElementById('noteInput').value = existing?.note || '';
  evidenceIndex = 0;
  renderEvidence(row, seg);
}

function selectVisSegment(row, channel, seg, el) {
  closeRoiEditor();
  document.querySelectorAll('.seg.selected').forEach(n => n.classList.remove('selected'));
  el.classList.add('selected');
  selected = {kind: 'visibility', row, channel, seg};
  const existing = visDecisions.get(visKeyFor(row.signal_group_id, channel, seg));
  document.getElementById('stateField').style.display = 'none';
  document.getElementById('visibilityField').style.display = '';
  document.getElementById('selectedGroup').textContent = `${row.signal_group_id} (${channel})`;
  document.getElementById('selectedInterval').textContent =
    `${seg.start_sample_token} .. ${seg.end_sample_token}`;
  document.getElementById('visibilityInput').value = existing?.visibility || seg.visibility;
  document.getElementById('statusInput').value = existing?.review_status || 'fixed';
  document.getElementById('noteInput').value = existing?.note || '';
  evidenceIndex = 0;
  // Visibility segments carry no crops of their own; borrow same-channel
  // crops from any state segment overlapping this time range for context.
  const candidates = (row.segments || [])
    .filter(s => s.start_timestamp <= seg.end_timestamp && s.end_timestamp >= seg.start_timestamp)
    .flatMap(s => s.candidates || [])
    .filter(c => c.channel === channel);
  renderEvidence(row, {...seg, candidates});
}

function viewHash(cand, withToken) {
  if (!cand) return '';
  const parts = [`ch=${encodeURIComponent(cand.channel || '')}`, `t=${cand.timestamp}`];
  if (withToken && cand.token) parts.unshift(`token=${encodeURIComponent(cand.token)}`);
  return '#' + parts.join('&');
}

function viewLinkFor(view, label, cand) {
  const a = document.createElement('a');
  a.className = 'viewlink small';
  a.target = '_blank';
  a.textContent = label + ' \\u2192';
  a.href = LINKS[view] + viewHash(cand, view === 'frame_view');
  return a;
}

// Header links track the evidence frame on screen; with nothing selected they
// still open the views at their own first frame.
function setViewLinks(cand) {
  document.getElementById('frameLink').href =
    LINKS.frame_view + viewHash(cand, true);
  document.getElementById('mapLink').href =
    LINKS.map_view + viewHash(cand, false);
}

function renderEvidence(row, seg) {
  const panel = document.getElementById('cropPanel');
  const candidates = seg.candidates || [];
  panel.innerHTML = '';
    if (!candidates.length) {
      const empty = document.createElement('div');
      empty.className = 'hint';
      empty.textContent = `No crop candidates in channels: ${DATA.crop_channels}.`;
      panel.appendChild(empty);
      return;
    }
  evidenceIndex = Math.min(evidenceIndex, candidates.length - 1);
  const tabs = document.createElement('div');
  tabs.className = 'candidateTabs';
  candidates.forEach((cand, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = idx === evidenceIndex ? 'active' : '';
    btn.textContent = `${idx + 1} ${cand.channel || 'camera'}` + (roiDecisions.has(cand.token) ? ' *' : '');
    btn.title = `rank=${cand.rank_score} score=${cand.detector_score || '-'} area=${cand.box_area}`;
    btn.addEventListener('click', () => {
      evidenceIndex = idx;
      renderEvidence(row, seg);
    });
    tabs.appendChild(btn);
  });
  panel.appendChild(tabs);

  const cand = candidates[evidenceIndex];
  const link = document.createElement('a');
  link.href = cand.full_image;
  link.target = '_blank';
  const img = document.createElement('img');
  img.src = cand.src;
  img.alt = `${cand.channel} crop`;
  link.appendChild(img);
  panel.appendChild(link);

  const seconds = ((cand.timestamp - DATA.t0) / 1e6).toFixed(2);
  const meta = document.createElement('div');
  meta.className = 'candidateMeta';
  meta.textContent =
    `${cand.channel} t=${seconds}s rank=${cand.rank_score} ` +
    `det=${cand.detector_score || '-'} area=${cand.box_area} ` +
    `src=${cand.source_type || '-'} vis=${cand.visibility || '-'} ` +
    `state=${cand.state || '-'} raw=${cand.raw_state || '-'}` +
    (roiDecisions.has(cand.token) ? ' [ROI fixed]' : '');
  panel.appendChild(meta);

  const path = document.createElement('div');
  path.className = 'mono';
  path.textContent = cand.filename;
  panel.appendChild(path);

  // The companion views cover boxes this timeline cannot reach (anything
  // unmatched to a map RE), so jumping straight to this frame in them is the
  // fastest way to see what else was in the image.
  setViewLinks(cand);
  const jump = document.createElement('div');
  jump.className = 'viewlinks';
  jump.appendChild(viewLinkFor('frame_view', 'Frame view', cand));
  jump.appendChild(viewLinkFor('map_view', 'Map view', cand));
  panel.appendChild(jump);

  const roiBtn = document.createElement('button');
  roiBtn.type = 'button';
  roiBtn.style.marginTop = '8px';
  roiBtn.textContent = 'Edit ROI on this frame';
  roiBtn.addEventListener('click', () => openRoiEditor(row, seg, cand.channel, cand.token));
  panel.appendChild(roiBtn);
}

function openRoiEditor(row, seg, channel, initialToken) {
  const frames = mergedRoiFrames(row, seg, channel);
  if (!frames.length) return;
  let idx = frames.findIndex(f => f.token === initialToken);
  if (idx < 0) idx = 0;
  roiState = {row, seg, channel, frames, idx, box: null, origBox: null, frame: null,
    img: null, imgW: 0, imgH: 0, win: null, drag: null};
  document.getElementById('roiEditor').style.display = '';
  loadRoiFrame();
}

function closeRoiEditor() {
  roiState = null;
  document.getElementById('roiEditor').style.display = 'none';
}

function loadRoiFrame() {
  const frame = roiState.frames[roiState.idx];
  const existing = roiDecisions.get(frame.token);
  roiState.frame = frame;
  roiState.box = (existing ? existing.box2d : frame.box2d).slice();
  roiState.origBox = frame.box2d.slice();
  document.getElementById('roiStatusInput').value = existing?.review_status || 'fixed';
  document.getElementById('roiNoteInput').value = existing?.note || '';
  const seg = roiState.seg;
  const outside = frame.timestamp != null && seg
    && (frame.timestamp < seg.start_timestamp || frame.timestamp > seg.end_timestamp);
  document.getElementById('roiFrameInfo').textContent =
    `${roiState.idx + 1} / ${roiState.frames.length}`
    + (existing ? ' [edited]' : '') + (outside ? ' [outside segment]' : '');
  const img = new Image();
  img.onload = () => {
    if (roiState.frame !== frame) return;
    roiState.img = img;
    roiState.imgW = img.naturalWidth;
    roiState.imgH = img.naturalHeight;
    roiState.win = cropWindow(roiState.box, roiState.imgW, roiState.imgH, 3.0, 60);
    syncRoiInputs();
    drawRoiCanvas();
  };
  img.src = frame.full_image;
}

function syncRoiInputs() {
  const [x0, y0, x1, y1] = roiState.box;
  document.getElementById('roiX0').value = x0.toFixed(1);
  document.getElementById('roiY0').value = y0.toFixed(1);
  document.getElementById('roiX1').value = x1.toFixed(1);
  document.getElementById('roiY1').value = y1.toFixed(1);
}

function onRoiNumberChange() {
  if (!roiState || !roiState.img) return;
  const x0 = parseFloat(document.getElementById('roiX0').value);
  const y0 = parseFloat(document.getElementById('roiY0').value);
  const x1 = parseFloat(document.getElementById('roiX1').value);
  const y1 = parseFloat(document.getElementById('roiY1').value);
  if ([x0, y0, x1, y1].some(v => !Number.isFinite(v))) return;
  roiState.box = clampBox(
    [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)],
    roiState.imgW, roiState.imgH
  );
  drawRoiCanvas();
}

// Keep one backing-store pixel per CSS pixel: the panel is resizable, and a
// mismatch would both blur the image and scale the 8px handle hit-boxes away
// from where they are drawn.
function syncRoiCanvasSize(canvas) {
  const size = Math.max(Math.round(canvas.clientWidth), 200);
  if (canvas.width !== size) {
    canvas.width = size;
    canvas.height = size;
  }
}

function drawRoiCanvas() {
  const canvas = document.getElementById('roiCanvas');
  syncRoiCanvasSize(canvas);
  const ctx = canvas.getContext('2d');
  const win = roiState.win;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(roiState.img, win[0], win[1], win[2] - win[0], win[3] - win[1],
    0, 0, canvas.width, canvas.height);
  const [cx0, cy0] = imageToCanvas([roiState.box[0], roiState.box[1]], win, canvas.width, canvas.height);
  const [cx1, cy1] = imageToCanvas([roiState.box[2], roiState.box[3]], win, canvas.width, canvas.height);
  ctx.strokeStyle = '#ffdd33';
  ctx.lineWidth = 2;
  ctx.strokeRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
  ctx.fillStyle = '#ffdd33';
  for (const h of roiHandles(cx0, cy0, cx1, cy1)) {
    ctx.fillRect(h.x - 4, h.y - 4, 8, 8);
  }
}

function roiPointerPos(ev, canvas) {
  const rect = canvas.getBoundingClientRect();
  return [
    (ev.clientX - rect.left) * (canvas.width / rect.width),
    (ev.clientY - rect.top) * (canvas.height / rect.height),
  ];
}

function roiBoxCanvasRect() {
  const canvas = document.getElementById('roiCanvas');
  const [cx0, cy0] = imageToCanvas([roiState.box[0], roiState.box[1]], roiState.win, canvas.width, canvas.height);
  const [cx1, cy1] = imageToCanvas([roiState.box[2], roiState.box[3]], roiState.win, canvas.width, canvas.height);
  return [cx0, cy0, cx1, cy1];
}

function roiOnPointerDown(ev) {
  if (!roiState || !roiState.img) return;
  const canvas = document.getElementById('roiCanvas');
  const [px, py] = roiPointerPos(ev, canvas);
  const [cx0, cy0, cx1, cy1] = roiBoxCanvasRect();
  const mode = roiHitTest(px, py, cx0, cy0, cx1, cy1);
  if (!mode) return;
  roiState.drag = {mode, startPx: px, startPy: py, startBox: roiState.box.slice()};
  ev.preventDefault();
}

function roiOnPointerMove(ev) {
  if (!roiState || !roiState.img) return;
  const canvas = document.getElementById('roiCanvas');
  const [px, py] = roiPointerPos(ev, canvas);
  if (!roiState.drag) {
    const [cx0, cy0, cx1, cy1] = roiBoxCanvasRect();
    const mode = roiHitTest(px, py, cx0, cy0, cx1, cy1);
    canvas.style.cursor = ROI_CURSORS[mode] || 'default';
    return;
  }
  const win = roiState.win;
  const [ix, iy] = canvasToImage([px, py], win, canvas.width, canvas.height);
  const {mode, startBox} = roiState.drag;
  let [x0, y0, x1, y1] = startBox;
  if (mode === 'move') {
    const [sx, sy] = canvasToImage([roiState.drag.startPx, roiState.drag.startPy], win, canvas.width, canvas.height);
    const dx = ix - sx, dy = iy - sy;
    x0 += dx; x1 += dx; y0 += dy; y1 += dy;
    if (x0 < 0) { x1 -= x0; x0 = 0; }
    if (y0 < 0) { y1 -= y0; y0 = 0; }
    if (x1 > roiState.imgW) { x0 -= (x1 - roiState.imgW); x1 = roiState.imgW; }
    if (y1 > roiState.imgH) { y0 -= (y1 - roiState.imgH); y1 = roiState.imgH; }
  } else {
    if (mode.includes('w')) x0 = ix;
    if (mode.includes('e')) x1 = ix;
    if (mode.includes('n')) y0 = iy;
    if (mode.includes('s')) y1 = iy;
  }
  roiState.box = clampBox(
    [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)],
    roiState.imgW, roiState.imgH
  );
  syncRoiInputs();
  drawRoiCanvas();
}

function roiOnPointerUp() {
  if (roiState) roiState.drag = null;
}

function roiOnWheel(ev) {
  if (!roiState || !roiState.img) return;
  ev.preventDefault();
  const canvas = document.getElementById('roiCanvas');
  const [px, py] = roiPointerPos(ev, canvas);
  const win = roiState.win;
  const [ix, iy] = canvasToImage([px, py], win, canvas.width, canvas.height);
  const curSize = Math.max(win[2] - win[0], win[3] - win[1]);
  const factor = Math.exp(ev.deltaY * 0.001);
  const minSize = 20;
  const maxSize = Math.max(roiState.imgW, roiState.imgH);
  const size = Math.min(Math.max(curSize * factor, minSize), maxSize);
  const fx = (ix - win[0]) / Math.max(win[2] - win[0], 1e-6);
  const fy = (iy - win[1]) / Math.max(win[3] - win[1], 1e-6);
  let x0 = ix - fx * size, y0 = iy - fy * size;
  let x1 = x0 + size, y1 = y0 + size;
  if (x0 < 0) { x1 -= x0; x0 = 0; }
  if (y0 < 0) { y1 -= y0; y0 = 0; }
  if (x1 > roiState.imgW) { x0 -= (x1 - roiState.imgW); x1 = roiState.imgW; }
  if (y1 > roiState.imgH) { y0 -= (y1 - roiState.imgH); y1 = roiState.imgH; }
  roiState.win = [Math.max(x0, 0), Math.max(y0, 0), Math.min(x1, roiState.imgW), Math.min(y1, roiState.imgH)];
  drawRoiCanvas();
}

function roiPrev() {
  if (!roiState) return;
  roiState.idx = (roiState.idx - 1 + roiState.frames.length) % roiState.frames.length;
  loadRoiFrame();
}

function roiNext() {
  if (!roiState) return;
  roiState.idx = (roiState.idx + 1) % roiState.frames.length;
  loadRoiFrame();
}

function roiRecenter() {
  if (!roiState || !roiState.img) return;
  roiState.win = cropWindow(roiState.box, roiState.imgW, roiState.imgH, 3.0, 60);
  drawRoiCanvas();
}

function roiReset() {
  if (!roiState) return;
  roiState.box = roiState.origBox.slice();
  syncRoiInputs();
  drawRoiCanvas();
}

// Adjacent frame's current box: its saved fix if one exists, else its
// detected box. Lets a reviewer drag a good box across a run of frames
// instead of redrawing it from scratch on each one.
function roiBoxForFrameIndex(idx) {
  const frame = roiState.frames[idx];
  if (!frame) return null;
  const existing = roiDecisions.get(frame.token);
  return (existing ? existing.box2d : frame.box2d).slice();
}

function roiCopyFromPrev() {
  if (!roiState || !roiState.img || roiState.frames.length < 2) return;
  const idx = (roiState.idx - 1 + roiState.frames.length) % roiState.frames.length;
  const box = roiBoxForFrameIndex(idx);
  if (!box) return;
  roiState.box = clampBox(box, roiState.imgW, roiState.imgH);
  syncRoiInputs();
  drawRoiCanvas();
}

function roiCopyFromNext() {
  if (!roiState || !roiState.img || roiState.frames.length < 2) return;
  const idx = (roiState.idx + 1) % roiState.frames.length;
  const box = roiBoxForFrameIndex(idx);
  if (!box) return;
  roiState.box = clampBox(box, roiState.imgW, roiState.imgH);
  syncRoiInputs();
  drawRoiCanvas();
}

function roiSave() {
  if (!roiState) return;
  const frame = roiState.frame;
  const decision = {
    annotation_token: frame.token,
    channel: frame.channel,
    sample_token: frame.sample_token,
    timestamp: frame.timestamp,
    box2d: roiState.box.map(v => Math.round(v * 10) / 10),
    review_status: document.getElementById('roiStatusInput').value,
    source: 'manual_timeline_review',
    note: document.getElementById('roiNoteInput').value,
    signal_group_id: roiState.row.signal_group_id,
  };
  roiDecisions.set(frame.token, decision);
  renderDecisionList();
  renderChart();
  scheduleAutoSave();
  loadRoiFrame();
}

function roiClear() {
  if (!roiState) return;
  roiDecisions.delete(roiState.frame.token);
  renderDecisionList();
  renderChart();
  scheduleAutoSave();
  loadRoiFrame();
}

function addDecision() {
  if (!selected) return;
  if (selected.kind === 'visibility') {
    const {row, channel, seg} = selected;
    const decision = {
      start_sample_token: seg.start_sample_token,
      end_sample_token: seg.end_sample_token,
      start_timestamp: seg.start_timestamp,
      end_timestamp: seg.end_timestamp,
      visibility: document.getElementById('visibilityInput').value,
      review_status: document.getElementById('statusInput').value,
      source: 'manual_timeline_review',
      n_frames: seg.n_frames,
      note: document.getElementById('noteInput').value,
      signal_group_id: row.signal_group_id,
      channel,
    };
    visDecisions.set(visKeyFor(row.signal_group_id, channel, seg), decision);
  } else {
    const {row, seg} = selected;
    const decision = {
      start_sample_token: seg.start_sample_token,
      end_sample_token: seg.end_sample_token,
      start_timestamp: seg.start_timestamp,
      end_timestamp: seg.end_timestamp,
      state: document.getElementById('stateInput').value.trim() || 'unknown',
      review_status: document.getElementById('statusInput').value,
      source: 'manual_timeline_review',
      n_frames: seg.n_frames,
      flags: seg.flags || [],
      note: document.getElementById('noteInput').value,
      signal_group_id: row.signal_group_id,
    };
    decisions.set(keyFor(row.signal_group_id, seg), decision);
  }
  renderDecisionList();
  renderChart();
  scheduleAutoSave();
}

function clearDecision() {
  if (!selected) return;
  if (selected.kind === 'visibility') {
    visDecisions.delete(visKeyFor(selected.row.signal_group_id, selected.channel, selected.seg));
  } else {
    decisions.delete(keyFor(selected.row.signal_group_id, selected.seg));
  }
  renderDecisionList();
  renderChart();
  scheduleAutoSave();
}

function buildPayload() {
  const byGroup = new Map();
  for (const row of DATA.rows) {
    byGroup.set(row.signal_group_id, {
      signal_group_id: row.signal_group_id,
      member_ways: row.member_ways,
      regulatory_element_ids: row.regulatory_element_ids,
      decisions: [],
      visibility_decisions: {},
      roi_decisions: [],
    });
  }
  for (const decision of decisions.values()) {
    const group = byGroup.get(decision.signal_group_id);
    if (!group) continue;
    const clean = {...decision};
    delete clean.signal_group_id;
    group.decisions.push(clean);
  }
  for (const decision of visDecisions.values()) {
    const group = byGroup.get(decision.signal_group_id);
    if (!group) continue;
    const clean = {...decision};
    delete clean.signal_group_id;
    delete clean.channel;
    (group.visibility_decisions[decision.channel] ||= []).push(clean);
  }
  for (const decision of roiDecisions.values()) {
    const group = byGroup.get(decision.signal_group_id);
    if (!group) continue;
    const clean = {...decision};
    delete clean.signal_group_id;
    group.roi_decisions.push(clean);
  }
  return {
    schema_version: 'traffic_signal_re_review/v1',
    source_timeseries: DATA.source_timeseries,
    created_at: new Date().toISOString(),
    groups: [...byGroup.values()].filter(
      g => g.decisions.length || Object.keys(g.visibility_decisions).length || g.roi_decisions.length
    ),
  };
}

function renderDecisionList() {
  const list = document.getElementById('decisionList');
  const payload = buildPayload();
  const nState = payload.groups.reduce((acc, g) => acc + g.decisions.length, 0);
  const nVis = payload.groups.reduce(
    (acc, g) => acc + Object.values(g.visibility_decisions).reduce((a, s) => a + s.length, 0), 0
  );
  const nRoi = payload.groups.reduce((acc, g) => acc + g.roi_decisions.length, 0);
  list.innerHTML =
    `<div class="hint">${nState} state decisions, ${nVis} visibility decisions, ${nRoi} ROI decisions staged</div>`;
  for (const group of payload.groups) {
    for (const d of group.decisions) {
      const item = document.createElement('div');
      item.className = 'decision';
      item.innerHTML = `<b>${group.signal_group_id}</b>${d.review_status}: ${d.state}<br>` +
        `<span class="mono">${d.start_sample_token} .. ${d.end_sample_token}</span>`;
      list.appendChild(item);
    }
    for (const [channel, segs] of Object.entries(group.visibility_decisions)) {
      for (const d of segs) {
        const item = document.createElement('div');
        item.className = 'decision';
        item.innerHTML = `<b>${group.signal_group_id} (${channel})</b>${d.review_status}: ${d.visibility}<br>` +
          `<span class="mono">${d.start_sample_token} .. ${d.end_sample_token}</span>`;
        list.appendChild(item);
      }
    }
    for (const d of group.roi_decisions) {
      const item = document.createElement('div');
      item.className = 'decision';
      item.innerHTML = `<b>${group.signal_group_id} (${d.channel})</b>${d.review_status}: box2d=[${d.box2d.join(', ')}]<br>` +
        `<span class="mono">${d.annotation_token} @ ${d.sample_token}</span>`;
      list.appendChild(item);
    }
  }
}

let autoSaveTimer = null;
let saveInFlight = false;
let saveQueued = false;

// Coalesce bursts of edits into one write, and never overlap two POSTs:
// a save landing out of order would leave the file behind the UI.
function scheduleAutoSave() {
  if (!SAVE_CONFIG) return;
  const box = document.getElementById('autoSave');
  if (!box.checked) return;
  clearTimeout(autoSaveTimer);
  autoSaveTimer = setTimeout(saveReview, 500);
}

async function postReview(endpoint) {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(buildPayload(), null, 2) + '\\n',
  });
  const body = await res.json();
  if (!res.ok || !body.ok) throw new Error(body.error || `HTTP ${res.status}`);
  return body;
}

// Auto-save only ever touches the draft. The committed file changes solely
// through the explicit Export/commit flow below.
async function saveReview() {
  if (!SAVE_CONFIG) return;
  if (saveInFlight) { saveQueued = true; return; }
  saveInFlight = true;
  const status = document.getElementById('saveStatus');
  status.textContent = 'saving draft...';
  try {
    const body = await postReview(SAVE_CONFIG.draft_endpoint);
    const at = new Date().toLocaleTimeString();
    status.textContent = `draft saved to ${body.path} at ${at}`;
  } catch (err) {
    status.textContent = `draft save failed: ${err.message}`;
  } finally {
    saveInFlight = false;
    if (saveQueued) { saveQueued = false; saveReview(); }
  }
}

function diffEntryHtml(cls, tag, entry) {
  const fmt = fields => Object.entries(fields)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `${k}=${Array.isArray(v) ? '[' + v.join(', ') + ']' : v}`)
    .join('  ') || '(empty)';
  let detail;
  if (cls === 'changed') {
    detail = `<span class="before">${escapeHtml(fmt(entry.before))}</span>`
      + ` &rarr; <span class="after">${escapeHtml(fmt(entry.after))}</span>`;
  } else if (cls === 'added') {
    detail = `<span class="after">${escapeHtml(fmt(entry.after))}</span>`;
  } else {
    detail = `<span class="before">${escapeHtml(fmt(entry.before))}</span>`;
  }
  return `<div class="diffEntry ${cls}"><span class="tag">${tag}</span>`
    + `<span class="mono">${escapeHtml(entry.label)}</span><br>${detail}</div>`;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

function renderDiff(diff, committedExists, path) {
  const body = document.getElementById('diffBody');
  document.getElementById('diffTarget').textContent =
    (committedExists ? 'updates ' : 'creates ') + path;
  if (!diff.total) {
    body.innerHTML = '<div class="hint">No differences: the committed file '
      + 'already matches what is staged.</div>';
    return;
  }
  const kinds = [['state', 'State'], ['visibility', 'Visibility'], ['roi', 'ROI']];
  let html = '';
  for (const [key, title] of kinds) {
    const section = diff[key];
    const n = section.added.length + section.removed.length + section.changed.length;
    if (!n) continue;
    html += `<div class="diffKind"><h3>${title} (${n})</h3>`;
    for (const e of section.added) html += diffEntryHtml('added', 'add', e);
    for (const e of section.changed) html += diffEntryHtml('changed', 'change', e);
    for (const e of section.removed) html += diffEntryHtml('removed', 'remove', e);
    html += '</div>';
  }
  body.innerHTML = html;
}

async function openCommitDialog() {
  if (!SAVE_CONFIG) return;
  const overlay = document.getElementById('diffOverlay');
  const body = document.getElementById('diffBody');
  const confirm = document.getElementById('diffConfirm');
  body.innerHTML = '<div class="hint">computing diff...</div>';
  document.getElementById('diffTarget').textContent = '';
  confirm.disabled = true;
  overlay.style.display = '';
  try {
    const res = await postReview(SAVE_CONFIG.diff_endpoint);
    renderDiff(res.diff, res.committed_exists, res.path);
    confirm.disabled = false;
  } catch (err) {
    body.innerHTML = `<div class="hint">diff failed: ${escapeHtml(err.message)}</div>`;
  }
}

function closeCommitDialog() {
  document.getElementById('diffOverlay').style.display = 'none';
}

async function confirmCommit() {
  const status = document.getElementById('saveStatus');
  const confirm = document.getElementById('diffConfirm');
  confirm.disabled = true;
  try {
    const body = await postReview(SAVE_CONFIG.commit_endpoint);
    const at = new Date().toLocaleTimeString();
    status.textContent = `committed to ${body.path} at ${at} (draft cleared)`;
    closeCommitDialog();
  } catch (err) {
    status.textContent = `commit failed: ${err.message}`;
    confirm.disabled = false;
  }
}

function exportReview() {
  const blob = new Blob([JSON.stringify(buildPayload(), null, 2) + '\\n'], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'traffic_signal_re_review.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

document.getElementById('addDecision').addEventListener('click', addDecision);
document.getElementById('clearDecision').addEventListener('click', clearDecision);
document.getElementById('exportReview').addEventListener('click', exportReview);
document.getElementById('roiPrev').addEventListener('click', roiPrev);
document.getElementById('roiNext').addEventListener('click', roiNext);
document.getElementById('roiSave').addEventListener('click', roiSave);
document.getElementById('roiReset').addEventListener('click', roiReset);
document.getElementById('roiCopyPrev').addEventListener('click', roiCopyFromPrev);
document.getElementById('roiCopyNext').addEventListener('click', roiCopyFromNext);
document.getElementById('roiRecenter').addEventListener('click', roiRecenter);
document.getElementById('roiClear').addEventListener('click', roiClear);
document.getElementById('roiClose').addEventListener('click', closeRoiEditor);
for (const id of ['roiX0', 'roiY0', 'roiX1', 'roiY1']) {
  document.getElementById(id).addEventListener('change', onRoiNumberChange);
}
const roiCanvasEl = document.getElementById('roiCanvas');
roiCanvasEl.addEventListener('mousedown', roiOnPointerDown);
roiCanvasEl.addEventListener('mousemove', roiOnPointerMove);
roiCanvasEl.addEventListener('wheel', roiOnWheel, {passive: false});
window.addEventListener('mouseup', roiOnPointerUp);
document.getElementById('importButton').addEventListener('click', () => {
  document.getElementById('importReview').click();
});
document.getElementById('importReview').addEventListener('change', ev => {
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    loadReview(JSON.parse(reader.result));
    renderDecisionList();
    renderChart();
    scheduleAutoSave();
  };
  reader.readAsText(file);
});

if (SAVE_CONFIG) {
  const btn = document.getElementById('commitReview');
  btn.style.display = '';
  btn.title = `review the diff, then write ${SAVE_CONFIG.commit_path}`;
  btn.addEventListener('click', openCommitDialog);
  document.getElementById('diffConfirm').addEventListener('click', confirmCommit);
  document.getElementById('diffCancel').addEventListener('click', closeCommitDialog);
  document.getElementById('diffOverlay').addEventListener('click', ev => {
    if (ev.target.id === 'diffOverlay') closeCommitDialog();
  });
  document.addEventListener('keydown', ev => {
    if (ev.key === 'Escape') closeCommitDialog();
  });
  document.getElementById('autoSaveRow').style.display = '';
  document.getElementById('autoSave').addEventListener('change', ev => {
    // Turning auto-save on flushes whatever is already staged, so the draft
    // matches the UI from that point on rather than from the next edit.
    if (ev.target.checked) scheduleAutoSave();
    else clearTimeout(autoSaveTimer);
  });
  const resumed = SAVE_CONFIG.resumed_from === 'draft'
    ? ' (resumed from draft)'
    : (SAVE_CONFIG.resumed_from === 'committed' ? ' (loaded committed file)' : '');
  document.getElementById('saveStatus').textContent =
    `draft: ${SAVE_CONFIG.draft_path}${resumed}`;
}

const ASIDE_DEFAULT_W = 430;
const ASIDE_MIN_W = 300;
const CHART_MIN_W = 320;

function setAsideWidth(px) {
  const max = Math.max(ASIDE_MIN_W, window.innerWidth - CHART_MIN_W);
  const w = Math.round(Math.min(Math.max(px, ASIDE_MIN_W), max));
  document.documentElement.style.setProperty('--aside-w', w + 'px');
  // The ROI canvas scales with the panel, so its backing store has to follow
  // or the image goes soft and the handle hit-boxes drift off-screen.
  if (roiState && roiState.img) drawRoiCanvas();
  return w;
}

(function initResizer() {
  const resizer = document.getElementById('resizer');
  const stored = parseFloat(localStorage.getItem('tlrAsideWidth'));
  if (Number.isFinite(stored)) setAsideWidth(stored);

  resizer.addEventListener('pointerdown', ev => {
    ev.preventDefault();
    resizer.setPointerCapture(ev.pointerId);
    resizer.classList.add('dragging');
    document.body.style.userSelect = 'none';
  });
  resizer.addEventListener('pointermove', ev => {
    if (!resizer.hasPointerCapture(ev.pointerId)) return;
    setAsideWidth(window.innerWidth - ev.clientX);
  });
  const endDrag = ev => {
    if (!resizer.hasPointerCapture(ev.pointerId)) return;
    resizer.releasePointerCapture(ev.pointerId);
    resizer.classList.remove('dragging');
    document.body.style.userSelect = '';
    const current = document.documentElement.style.getPropertyValue('--aside-w');
    localStorage.setItem('tlrAsideWidth', parseFloat(current) || ASIDE_DEFAULT_W);
  };
  resizer.addEventListener('pointerup', endDrag);
  resizer.addEventListener('pointercancel', endDrag);
  resizer.addEventListener('dblclick', () => {
    setAsideWidth(ASIDE_DEFAULT_W);
    localStorage.setItem('tlrAsideWidth', ASIDE_DEFAULT_W);
  });
  // Re-clamp when the window shrinks below what the stored width allows.
  window.addEventListener('resize', () => {
    const current = parseFloat(document.documentElement.style.getPropertyValue('--aside-w'));
    setAsideWidth(Number.isFinite(current) ? current : ASIDE_DEFAULT_W);
  });
})();

setViewLinks(null);
loadReview(INITIAL_REVIEW || {});
renderDecisionList();
renderChart();
</script></body></html>
"""
    save_config = json.dumps(
        {
            "draft_endpoint": DRAFT_ENDPOINT,
            "commit_endpoint": COMMIT_ENDPOINT,
            "diff_endpoint": DIFF_ENDPOINT,
            "draft_path": os.path.relpath(draft_out, root),
            "commit_path": os.path.relpath(review_out, root),
            "resumed_from": resumed_from,
        }
        if args.serve else None
    )
    links = companion_links(
        root, output_path.parent, frame_view=args.frame_view, map_view=args.map_view
    )
    page = (
        page.replace("__SUBTITLE__", subtitle)
        .replace("__LINKS__", json.dumps(links, ensure_ascii=False))
        .replace("__DATA__", data_json)
        .replace("__REVIEW__", review_json)
        .replace("__SAVE_CONFIG__", save_config)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    print(f"wrote {output_path} ({len(rows)} groups)")

    if args.serve:
        if resumed_from == "draft":
            print(f"resuming from draft {draft_out}", flush=True)
        serve_review(root, output_path, review_out, draft_out, args.port)


if __name__ == "__main__":
    main()
