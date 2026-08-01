"""Legacy object_ann -> B/B' normalization (REFACTOR_PLAN.md phase 4).

Used by export_object_ann_to_t4dataset.py: takes an existing T4 dataset whose
traffic-light annotations are already `object_ann.json` but whose map
identity still lives in the deprecated repo-local
`traffic_light_map_association.json`, and normalizes it into B/B' (splitting
one 2D instance into several when it maps to more than one map primitive,
and rebuilding traffic_light.json from the resolved associations).
"""

from __future__ import annotations

import json
from pathlib import Path

from tlr_autolabel.core.io import load_json, token_of
from tlr_autolabel.t4.traffic_light import build_traffic_light_row

GENERATED = {
    "object_ann.json",
    "category.json",
    "attribute.json",
    "instance.json",
    "traffic_light.json",
    "traffic_light_instance_map.json",
    "traffic_light_map_association.json",
}


def write_json(path: Path, data: object) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def symlink_dataset(src: Path, out: Path) -> Path:
    src_ann = src / "annotation"
    ann_out = out / "annotation"
    ann_out.mkdir(parents=True, exist_ok=True)

    for entry in src.iterdir():
        if entry.name == "annotation":
            continue
        dst = out / entry.name
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(entry)

    for entry in src_ann.iterdir():
        if entry.name in GENERATED:
            continue
        dst = ann_out / entry.name
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(entry)

    return ann_out


def association_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    by_object_ann: dict[str, set[str]] = {}
    for row in load_json(path):
        object_ann_token = row.get("object_ann_token")
        map_id = (
            row.get("primitive_id")
            or row.get("map_traffic_light_id")
            or row.get("traffic_light_id")
        )
        if not object_ann_token or not map_id:
            continue
        by_object_ann.setdefault(str(object_ann_token), set()).add(str(map_id))

    ambiguous = {k: sorted(v) for k, v in by_object_ann.items() if len(v) > 1}
    if ambiguous:
        examples = ", ".join(
            f"{k}:{'/'.join(v)}" for k, v in list(ambiguous.items())[:5]
        )
        raise SystemExit(
            "one object_ann_token maps to multiple primitive_id "
            f"values ({examples})"
        )
    return {k: next(iter(v)) for k, v in by_object_ann.items()}


def sample_data_timestamps(annotation_dir: Path) -> dict[str, int]:
    path = annotation_dir / "sample_data.json"
    if not path.exists():
        return {}
    return {row["token"]: int(row.get("timestamp", 0)) for row in load_json(path)}


def scene_name(annotation_dir: Path) -> str:
    path = annotation_dir / "scene.json"
    if not path.exists():
        return "scene"
    scenes = load_json(path)
    return scenes[0].get("name") or "scene" if scenes else "scene"


def split_instance_token(original_token: str, map_id: str) -> str:
    return token_of("split_instance", original_token, map_id)


def observation_instance_token(object_ann_token: str) -> str:
    return token_of("object_ann_instance", object_ann_token)


def normalize_object_ann(
    object_ann: list[dict],
    association_by_object_ann: dict[str, str],
) -> tuple[list[dict], dict[str, str], dict[str, set[str]], int, int]:
    tokens = {row["token"] for row in object_ann}
    missing_refs = sorted(set(association_by_object_ann) - tokens)
    if missing_refs:
        examples = ", ".join(missing_refs[:5])
        raise SystemExit(
            "traffic_light_map_association.json references missing object_ann "
            f"tokens ({examples})"
        )

    original_instance_by_object_ann: dict[str, str] = {}
    map_ids_by_instance: dict[str, set[str]] = {}
    generated_missing_instances = 0
    for row in object_ann:
        token = row["token"]
        instance_token = row.get("instance_token")
        if not instance_token:
            instance_token = observation_instance_token(token)
            generated_missing_instances += 1
        instance_token = str(instance_token)
        original_instance_by_object_ann[token] = instance_token
        map_id = association_by_object_ann.get(token)
        if map_id:
            map_ids_by_instance.setdefault(instance_token, set()).add(map_id)

    split_instances = {
        instance_token
        for instance_token, map_ids in map_ids_by_instance.items()
        if len(map_ids) > 1
    }

    normalized: list[dict] = []
    normalized_instance_by_object_ann: dict[str, str] = {}
    split_rows = 0
    for row in object_ann:
        token = row["token"]
        original_instance = original_instance_by_object_ann[token]
        map_id = association_by_object_ann.get(token)
        instance_token = original_instance
        if original_instance in split_instances and map_id:
            instance_token = split_instance_token(original_instance, map_id)
            split_rows += 1
        new_row = dict(row)
        new_row["instance_token"] = instance_token
        normalized.append(new_row)
        normalized_instance_by_object_ann[token] = instance_token

    relation_map_ids_by_instance: dict[str, set[str]] = {}
    for object_ann_token, map_id in association_by_object_ann.items():
        instance_token = normalized_instance_by_object_ann[object_ann_token]
        relation_map_ids_by_instance.setdefault(instance_token, set()).add(map_id)
    ambiguous = {
        k: sorted(v) for k, v in relation_map_ids_by_instance.items() if len(v) > 1
    }
    if ambiguous:
        examples = ", ".join(
            f"{k}:{'/'.join(v)}" for k, v in list(ambiguous.items())[:5]
        )
        raise SystemExit(
            "one instance_token maps to multiple primitive_id "
            f"values after normalization ({examples})"
        )

    return (
        normalized,
        normalized_instance_by_object_ann,
        relation_map_ids_by_instance,
        generated_missing_instances,
        split_rows,
    )


def rebuild_instances(
    source_instances: list[dict],
    object_ann: list[dict],
    sample_annotation: list[dict],
    timestamps_by_sample_data: dict[str, int],
    scene: str,
) -> list[dict]:
    source_by_token = {row["token"]: row for row in source_instances}
    sample_annotation_instance_tokens = {
        row["instance_token"] for row in sample_annotation if row.get("instance_token")
    }
    object_ann_by_instance: dict[str, list[tuple[tuple[int, int], dict]]] = {}
    for index, row in enumerate(object_ann):
        instance_token = row.get("instance_token")
        if not instance_token:
            continue
        timestamp = timestamps_by_sample_data.get(row.get("sample_data_token"), 0)
        object_ann_by_instance.setdefault(str(instance_token), []).append(
            ((timestamp, index), row)
        )

    final_instances: list[dict] = []
    seen: set[str] = set()
    for row in source_instances:
        if row["token"] in sample_annotation_instance_tokens:
            final_instances.append(row)
            seen.add(row["token"])

    for instance_token in sorted(object_ann_by_instance):
        rows = [row for _, row in sorted(object_ann_by_instance[instance_token])]
        source = dict(source_by_token.get(instance_token, {}))
        source["token"] = instance_token
        source["category_token"] = rows[0].get("category_token") or source.get(
            "category_token"
        )
        if not source.get("instance_name"):
            source["instance_name"] = f"{scene}::{instance_token[:8]}"
        source["nbr_annotations"] = len(rows)
        source["first_annotation_token"] = rows[0]["token"]
        source["last_annotation_token"] = rows[-1]["token"]
        if instance_token not in seen:
            final_instances.append(source)
            seen.add(instance_token)

    return final_instances


def build_traffic_light(
    association_by_object_ann: dict[str, str],
    instance_by_object_ann: dict[str, str],
) -> list[dict]:
    pairs = sorted(
        {
            (instance_by_object_ann[object_ann_token], map_id)
            for object_ann_token, map_id in association_by_object_ann.items()
        }
    )
    return [build_traffic_light_row(instance_token, map_id) for instance_token, map_id in pairs]
