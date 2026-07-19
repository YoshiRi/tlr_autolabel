#!/usr/bin/env python3
"""Export a camera slice from this T4-style dataset as a CVAT signal task."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


SIGNAL_LABELS = {"traffic_light"}


def load_json(path: Path):
    return json.loads(path.read_text())


def indent(element: ET.Element, level: int = 0) -> None:
    space = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = space + "  "
        for child in element:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = space
    if level and (not element.tail or not element.tail.strip()):
        element.tail = space


def add_text(parent: ET.Element, name: str, value: object | None = "") -> ET.Element:
    child = ET.SubElement(parent, name)
    child.text = "" if value is None else str(value)
    return child


def add_label(parent: ET.Element, name: str, attrs: list[tuple[str, str, str, bool]]) -> None:
    label = ET.SubElement(parent, "label")
    add_text(label, "name", name)
    add_text(label, "type", "bbox")
    attributes = ET.SubElement(label, "attributes")
    for attr_name, input_type, values, mutable in attrs:
        attr = ET.SubElement(attributes, "attribute")
        add_text(attr, "name", attr_name)
        add_text(attr, "mutable", "True" if mutable else "False")
        add_text(attr, "input_type", input_type)
        default = values.splitlines()[0] if input_type == "select" and values else ""
        add_text(attr, "default_value", default)
        add_text(attr, "values", values)


def build_channel_index(root: Path):
    sensors = load_json(root / "annotation/sensor.json")
    calibrated = load_json(root / "annotation/calibrated_sensor.json")
    sample_data = load_json(root / "annotation/sample_data.json")

    sensor_by_token = {row["token"]: row for row in sensors}
    channel_by_calibrated = {
        row["token"]: sensor_by_token[row["sensor_token"]]["channel"] for row in calibrated
    }

    by_channel: dict[str, list[dict]] = defaultdict(list)
    for row in sample_data:
        channel = channel_by_calibrated[row["calibrated_sensor_token"]]
        by_channel[channel].append({**row, "channel": channel})

    for rows in by_channel.values():
        rows.sort(key=lambda row: (row["timestamp"], row["filename"]))
    return by_channel


def load_signal_annotations(path: Path | None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    if path is None or not path.exists():
        return grouped
    payload = load_json(path)
    annotations = payload.get("annotations", payload) if isinstance(payload, dict) else payload
    for ann in annotations:
        token = ann.get("sample_data_token")
        if token:
            grouped[token].append(ann)
    return grouped


def load_re_flags(path: Path | None) -> dict[tuple[str, str], dict]:
    """{(regulatory_element_id, sample_token): {priority, flags}} from the RE
    verification report, for injecting cross-camera/temporal review priority."""
    lookup: dict[tuple[str, str], dict] = {}
    if path is None or not path.exists():
        return lookup
    for fo in load_json(path).get("flagged_observations", []):
        lookup[(fo["regulatory_element_id"], fo["sample_token"])] = {
            "priority": fo.get("review_priority", 0.0),
            "flags": fo.get("flags", []),
        }
    return lookup


def box_triage(ann: dict, re_flags: dict[tuple[str, str], dict]) -> tuple[float, list[str]]:
    """Per-box review priority (higher = review sooner) + human-readable reasons.
    Combines box-local signals with the RE report's cross-camera/temporal flags,
    so a reviewer can filter/sort straight to what's worth checking."""
    attrs = ann.get("attributes") or {}
    priority, reasons = 0.0, []

    if not attrs.get("map_traffic_light_id"):
        priority += 1.0
        reasons.append("unmatched")            # real signal missing from map, or a FP?

    try:
        score = float(attrs.get("detector_score", "") or "nan")
    except ValueError:
        score = float("nan")
    if score == score and score < 0.7:         # not NaN and low
        priority += (0.7 - score) * 2.0
        reasons.append(f"low_score:{score:.2f}")

    if (attrs.get("state") or "unknown") == "unknown":
        priority += 0.5
        reasons.append("state_unknown")

    sample_token = ann.get("sample_token", "")
    for re_id in (attrs.get("regulatory_element_id") or "").split(","):
        hit = re_flags.get((re_id.strip(), sample_token))
        if hit:
            priority += hit["priority"]
            reasons += hit["flags"]
            break

    return round(priority, 2), sorted(set(reasons))


def cvat_image_name(row: dict) -> str:
    path = Path(row["filename"])
    channel = row.get("channel") or (path.parts[1] if len(path.parts) > 1 and path.parts[0] == "data" else "image")
    return str(Path("images", f"{channel}_{path.name}"))


def add_meta(root_el: ET.Element, task_name: str, size: int) -> None:
    meta = ET.SubElement(root_el, "meta")
    task = ET.SubElement(meta, "task")
    add_text(task, "id", 0)
    add_text(task, "name", task_name)
    add_text(task, "size", size)
    add_text(task, "mode", "annotation")
    add_text(task, "overlap", 0)
    add_text(task, "bugtracker", "")
    add_text(task, "flipped", "False")
    now = datetime.now(timezone.utc).isoformat()
    add_text(task, "created", now)
    add_text(task, "updated", now)

    # Attribute contract: docs/cvat_interop.md (traffic_signal_2d/v2).
    # select defaults = first line (applies to newly drawn boxes in CVAT).
    labels = ET.SubElement(task, "labels")
    add_label(
        labels,
        "traffic_light",
        [
            ("state", "text", "", True),
            ("signal_kind", "select", "vehicle\npedestrian\nother\nunknown", True),
            ("visibility", "select", "unknown\nclear\npartial_occluded\nheavy_occluded", True),
            ("review_status", "select", "unchecked\naccepted\nrejected\nfixed", True),
            ("map_traffic_light_id", "text", "", True),
            ("regulatory_element_id", "text", "", False),
            ("facing", "text", "", False),
            ("raw_state", "text", "", False),
            ("detector_score", "text", "", False),
            ("source_type", "select", "manual\nprojected_map\nauto", False),
            ("annotation_uid", "text", "", False),
            # triage aids (read at export time; ignored on import). Filter in
            # CVAT with e.g. review_priority > 1.5 to jump to only suspicious boxes.
            ("review_priority", "number", "0;100;0.01", False),
            ("flags", "text", "", False),
        ],
    )
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    add_text(segment, "id", 0)
    add_text(segment, "start", 0)
    add_text(segment, "stop", max(size - 1, 0))
    add_text(segment, "url", "")
    owner = ET.SubElement(task, "owner")
    add_text(owner, "username", "")
    add_text(owner, "email", "")
    add_text(meta, "dumped", now)


def add_box(image_el: ET.Element, ann: dict, re_flags: dict[tuple[str, str], dict]) -> float:
    box = ann.get("box2d") or ann.get("bbox")
    if not box or len(box) != 4:
        return 0.0
    label = ann.get("label", "traffic_light")
    if label not in SIGNAL_LABELS:
        return 0.0
    attrs = dict(ann.get("attributes") or {})
    priority, reasons = box_triage(ann, re_flags)
    attrs["review_priority"] = f"{priority:.2f}"
    attrs["flags"] = ",".join(reasons)
    box_el = ET.SubElement(
        image_el,
        "box",
        {
            "label": label,
            "source": ann.get("source", "manual"),
            "xtl": f"{float(box[0]):.2f}",
            "ytl": f"{float(box[1]):.2f}",
            "xbr": f"{float(box[2]):.2f}",
            "ybr": f"{float(box[3]):.2f}",
            "occluded": "1" if ann.get("occluded") else "0",
            "z_order": str(int(ann.get("z_order", 0))),
        },
    )
    if ann.get("token") and "annotation_uid" not in attrs:
        attrs = {**attrs, "annotation_uid": ann["token"]}
    for key, value in attrs.items():
        attr = ET.SubElement(box_el, "attribute", {"name": key})
        attr.text = "" if value is None else str(value)
    return priority


def frame_priority(row: dict, signal_by_sample_data, re_flags) -> float:
    """Max box review-priority in a frame (for --worst-first / --min-priority)."""
    return max((box_triage(a, re_flags)[0]
                for a in signal_by_sample_data.get(row["token"], [])), default=0.0)


def build_xml(task_name: str, rows: list[dict], signal_by_sample_data: dict[str, list[dict]],
              re_flags: dict[tuple[str, str], dict]) -> ET.ElementTree:
    root_el = ET.Element("annotations")
    add_text(root_el, "version", "1.1")
    add_meta(root_el, task_name, len(rows))
    # CVAT assigns frame numbers by sorting the uploaded images by name and uses
    # the XML <image id> as the frame index. So `id` MUST equal the name-sorted
    # position, or boxes land on the wrong frames (they appear shifted). We sort
    # by CVAT image name here to guarantee that invariant regardless of the
    # order `rows` arrived in.
    rows = sorted(rows, key=cvat_image_name)
    for image_id, row in enumerate(rows):
        image_el = ET.SubElement(
            root_el,
            "image",
            {
                "id": str(image_id),
                "name": cvat_image_name(row),
                "width": str(row["width"]),
                "height": str(row["height"]),
            },
        )
        for ann in signal_by_sample_data.get(row["token"], []):
            add_box(image_el, ann, re_flags)
    indent(root_el)
    return ET.ElementTree(root_el)


def write_zip(root: Path, output: Path, rows: list[dict], xml_tree: ET.ElementTree, include_images: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        xml_path = tmp_path / "annotations.xml"
        xml_tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(xml_path, "annotations.xml")
            if include_images:
                for row in rows:
                    source = root / row["filename"]
                    if not source.exists():
                        raise FileNotFoundError(source)
                    archive.write(source, cvat_image_name(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--count", default=20, type=int, help="Use 0 or a negative value for all remaining frames.")
    parser.add_argument("--signal-ann", type=Path, default=Path("annotation/traffic_signal_2d_ann.json"))
    parser.add_argument("--re-report", type=Path,
                        default=Path("build/tl_match/re_verification_report.json"),
                        help="RE verification report for cross-camera/temporal review priority")
    parser.add_argument("--min-priority", type=float, default=None,
                        help="keep only frames whose max box review_priority >= this "
                             "(builds a focused suspicious-frames review task). "
                             "To review worst-first inside CVAT, filter on the "
                             "review_priority attribute — frame order stays temporal "
                             "so the id<->image mapping CVAT needs stays correct.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    by_channel = build_channel_index(dataset_root)
    if args.camera not in by_channel:
        raise SystemExit(f"unknown camera/channel: {args.camera}")

    rows = by_channel[args.camera]
    if args.start < 0 or args.start >= len(rows):
        raise SystemExit(f"--start out of range: {args.start}")
    end = len(rows) if args.count <= 0 else min(args.start + args.count, len(rows))
    selected = rows[args.start:end]

    signal_path = args.signal_ann
    if not signal_path.is_absolute():
        signal_path = dataset_root / signal_path
    signal_by_sample_data = load_signal_annotations(signal_path)
    re_path = args.re_report if args.re_report.is_absolute() else dataset_root / args.re_report
    re_flags = load_re_flags(re_path)

    if args.min_priority is not None:
        selected = [r for r in selected
                    if frame_priority(r, signal_by_sample_data, re_flags) >= args.min_priority]
        if not selected:
            raise SystemExit(f"no frames with review_priority >= {args.min_priority}")

    suffix = f"_p{args.min_priority:g}" if args.min_priority is not None else ""
    task_name = f"{args.camera}_{args.start:06d}_{end - 1:06d}{suffix}"
    output = args.output or dataset_root / "build/cvat_signal" / f"{task_name}.zip"

    xml_tree = build_xml(task_name, selected, signal_by_sample_data, re_flags)
    write_zip(dataset_root, output, selected, xml_tree, include_images=not args.no_images)
    print(f"wrote {output}")
    print(f"camera={args.camera} frames_in_task={len(selected)} "
          f"(from {args.start}..{end - 1}"
          + (f", min_priority={args.min_priority}" if args.min_priority is not None else "")
          + ")")


if __name__ == "__main__":
    main()
