#!/usr/bin/env python3
"""Import reviewed CVAT signal XML into the Tier B sidecar (traffic_signal_2d/v2).

Contract: docs/cvat_interop.md. Validation is fail-loud:
  - every `state` token must parse against the canonical grammar (legacy
    paren-style tokens are normalized); unknown tokens abort with a list,
  - `map_traffic_light_id` must exist in the lanelet2 map (or be empty),
  - `regulatory_element_id` is always re-derived from the map,
  - `signal_kind` is derived from `state` when empty,
  - boxes without `annotation_uid` (drawn in CVAT) get a fresh token and
    default `source_type=manual`.
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from state_tokens import CANON_RE, LEGACY_RE, elements_key, parse_state
from tlr_autolabel.core.io import load_json

SIGNAL_LABELS = {"traffic_light"}
TOKEN_NAMESPACE = uuid.UUID("5e3a5a36-739d-4c1e-aaf4-bdfd8bb18857")


# ------------------------------------------------------------------ T4 lookup


def build_sample_data_index(root: Path) -> dict[str, dict]:
    sensors = load_json(root / "annotation/sensor.json")
    calibrated = load_json(root / "annotation/calibrated_sensor.json")
    sample_data = load_json(root / "annotation/sample_data.json")

    sensor_by_token = {row["token"]: row for row in sensors}
    channel_by_calibrated = {
        row["token"]: sensor_by_token[row["sensor_token"]]["channel"] for row in calibrated
    }

    by_filename: dict[str, dict] = {}
    for row in sample_data:
        channel = channel_by_calibrated[row["calibrated_sensor_token"]]
        enriched = {**row, "channel": channel}
        filename = row["filename"].lstrip("/")
        cvat_filename = str(Path("images") / f"{channel}_{Path(filename).name}")
        for key in (filename, "/" + filename, cvat_filename, "/" + cvat_filename):
            by_filename[key] = enriched
    return by_filename


def resolve_sample_data(image_name: str, index: dict[str, dict]) -> dict | None:
    normalized = image_name.lstrip("/")
    candidates = [normalized, "/" + normalized]
    if normalized.startswith("images/"):
        without = normalized.removeprefix("images/")
        candidates += [without, "/" + without,
                       str(Path("data") / without), "/" + str(Path("data") / without)]
    for cand in candidates:
        if cand in index:
            return index[cand]
    return None


# ---------------------------------------------------------------- lanelet2 map


def load_map_ids(osm_path: Path):
    """Return (traffic_light_way_ids, regulatory_by_way)."""
    root = ET.parse(osm_path).getroot()
    way_ids: set[str] = set()
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if tags.get("type") == "traffic_light":
            way_ids.add(way.get("id"))
    regulatory_by_way: dict[str, list[str]] = defaultdict(list)
    for rel in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
        if tags.get("type") != "regulatory_element" or tags.get("subtype") != "traffic_light":
            continue
        for member in rel.findall("member"):
            if member.get("role") == "refers" and member.get("type") == "way":
                regulatory_by_way[member.get("ref")].append(rel.get("id"))
    return way_ids, regulatory_by_way


# ------------------------------------------------------------------ validation


def invalid_state_tokens(state: str) -> list[str]:
    bad = []
    for token in filter(None, (t.strip() for t in (state or "").split(","))):
        if token == "unknown":
            continue
        if not (CANON_RE.match(token) or LEGACY_RE.match(token)):
            bad.append(token)
    return bad


def derive_signal_kind(elements: list[dict]) -> str:
    if any(e["shape"] == "ped" for e in elements):
        return "pedestrian"
    return "vehicle" if elements else "unknown"


# ---------------------------------------------------------------------- import


def parse_attrs(box_el: ET.Element) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for attr in box_el.findall("attribute"):
        name = attr.attrib.get("name")
        if name:
            attrs[name] = (attr.text or "").strip()
    return attrs


def stable_token(image_name: str, box_index: int, box: list[float]) -> str:
    key = json.dumps({"image": image_name, "i": box_index,
                      "box": [round(v, 3) for v in box]}, sort_keys=True)
    return uuid.uuid5(TOKEN_NAMESPACE, key).hex


def convert(xml_path: Path, dataset_root: Path) -> dict:
    index = build_sample_data_index(dataset_root)
    way_ids, regulatory_by_way = load_map_ids(dataset_root / "map/lanelet2_map.osm")
    root_el = ET.parse(xml_path).getroot()

    annotations = []
    errors: list[str] = []

    for image_el in root_el.findall("image"):
        image_name = image_el.attrib["name"]
        sample_data = resolve_sample_data(image_name, index)
        if sample_data is None:
            errors.append(f"{image_name}: not found in sample_data.json")
            continue
        for box_index, box_el in enumerate(image_el.findall("box")):
            if box_el.attrib.get("label", "") not in SIGNAL_LABELS:
                continue
            box = [float(box_el.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")]
            attrs = parse_attrs(box_el)
            where = f"{image_name}#box{box_index}"

            state_in = attrs.get("state", "")
            bad = invalid_state_tokens(state_in)
            if bad:
                errors.append(f"{where}: invalid state token(s) {bad} in {state_in!r}")
                continue
            elements = parse_state(state_in)
            state = elements_key(elements) or "unknown"

            way_id = attrs.get("map_traffic_light_id", "")
            if way_id and way_id not in way_ids:
                errors.append(f"{where}: map_traffic_light_id {way_id!r} not in lanelet2 map")
                continue

            token = attrs.get("annotation_uid", "")
            is_new = not token
            if is_new:
                token = stable_token(image_name, box_index, box)

            annotations.append(
                {
                    "token": token,
                    "sample_token": sample_data["sample_token"],
                    "sample_data_token": sample_data["token"],
                    "channel": sample_data["channel"],
                    "filename": sample_data["filename"],
                    "timestamp": sample_data["timestamp"],
                    "label": "traffic_light",
                    "box2d": box,
                    "occluded": box_el.attrib.get("occluded", "0") == "1",
                    "z_order": int(box_el.attrib.get("z_order", "0")),
                    "attributes": {
                        "state": state,
                        "signal_kind": attrs.get("signal_kind") or derive_signal_kind(elements),
                        "visibility": attrs.get("visibility") or "unknown",
                        "review_status": attrs.get("review_status") or "unchecked",
                        "map_traffic_light_id": way_id,
                        "regulatory_element_id": ",".join(regulatory_by_way.get(way_id, [])),
                        "map_candidate_id": attrs.get("map_candidate_id", ""),
                        "regulatory_element_id_candidate": attrs.get("regulatory_element_id_candidate", ""),
                        "unmatched_reason": attrs.get("unmatched_reason", ""),
                        "facing": attrs.get("facing", ""),
                        "raw_state": attrs.get("raw_state", ""),
                        "detector_score": attrs.get("detector_score", ""),
                        "source_type": attrs.get("source_type")
                                       or ("manual" if is_new else "auto"),
                        "temporal_source": attrs.get("temporal_source", ""),
                        "track_id": attrs.get("track_id", ""),
                        "tracking_status": attrs.get("tracking_status", ""),
                        "tracking_lost_frames": attrs.get("tracking_lost_frames", ""),
                    },
                }
            )

    if errors:
        detail = "\n  ".join(errors)
        raise SystemExit(f"import failed, {len(errors)} problem(s):\n  {detail}")

    return {
        "schema_version": "traffic_signal_2d/v2",
        "source": "cvat",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cvat_xml": str(xml_path),
        "annotations": annotations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument("--output", default=Path("annotation/traffic_signal_2d_ann.imported.json"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output if args.output.is_absolute() else dataset_root / args.output
    payload = convert(args.xml, dataset_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    counts = defaultdict(int)
    for a in payload["annotations"]:
        counts[a["attributes"]["review_status"]] += 1
    print(f"wrote {output}")
    print(f"signal boxes {len(payload['annotations'])} review_status={dict(counts)}")


if __name__ == "__main__":
    main()
