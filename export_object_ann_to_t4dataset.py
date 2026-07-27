#!/usr/bin/env python3
"""Normalize an existing object_ann-based T4 dataset into B/B'.

This adapter is for legacy/current datasets where traffic-light annotations are
already encoded as standard T4 `object_ann.json`, but map identity still lives in
the deprecated repo-local `traffic_light_map_association.json`.

Output:

  B:  object_ann/category/attribute/instance
  B': B + traffic_light.json when map relations exist

The source dataset is not edited. Large source folders and non-generated
annotation tables are symlinked into the derived dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GENERATED = {
    "object_ann.json",
    "category.json",
    "attribute.json",
    "instance.json",
    "traffic_light.json",
    "traffic_light_map_association.json",
}


def token_of(*parts: object) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


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
            row.get("traffic_light_linestring_id")
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
            "one object_ann_token maps to multiple traffic_light_linestring_id "
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
            "one instance_token maps to multiple traffic_light_linestring_id "
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
    return [
        {
            "token": token_of("traffic_light", instance_token, map_id),
            "instance_token": instance_token,
            "traffic_light_linestring_id": map_id,
        }
        for instance_token, map_id in pairs
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--map-association",
        default=Path("annotation/traffic_light_map_association.json"),
        type=Path,
        help="deprecated object_ann_token -> map id source",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.dataset_root.resolve()
    out = args.out.resolve()
    if out == src:
        raise SystemExit("--out must differ from --dataset-root")

    src_ann = src / "annotation"
    if not src_ann.exists():
        raise SystemExit(f"annotation directory not found: {src_ann}")

    ann_out = symlink_dataset(src, out)
    object_ann = load_json(src_ann / "object_ann.json")
    category = load_json(src_ann / "category.json")
    attribute = load_json(src_ann / "attribute.json")
    source_instances = load_json(src_ann / "instance.json")
    sample_annotation = (
        load_json(src_ann / "sample_annotation.json")
        if (src_ann / "sample_annotation.json").exists()
        else []
    )
    association_path = resolve_path(src, args.map_association)
    association_by_object_ann = association_map(association_path)

    (
        normalized_object_ann,
        instance_by_object_ann,
        relation_map_ids_by_instance,
        generated_missing_instances,
        split_rows,
    ) = normalize_object_ann(object_ann, association_by_object_ann)
    instances = rebuild_instances(
        source_instances,
        normalized_object_ann,
        sample_annotation,
        sample_data_timestamps(src_ann),
        scene_name(src_ann),
    )
    traffic_light = build_traffic_light(association_by_object_ann, instance_by_object_ann)

    instance_tokens = {row["token"] for row in instances}
    missing_relation_instances = [
        row["instance_token"]
        for row in traffic_light
        if row["instance_token"] not in instance_tokens
    ]
    if missing_relation_instances:
        raise SystemExit(
            "traffic_light.json would reference missing instance_token values: "
            + ", ".join(missing_relation_instances[:5])
        )
    ambiguous = {
        k: sorted(v) for k, v in relation_map_ids_by_instance.items() if len(v) > 1
    }
    if ambiguous:
        examples = ", ".join(
            f"{k}:{'/'.join(v)}" for k, v in list(ambiguous.items())[:5]
        )
        raise SystemExit(f"ambiguous traffic_light relations remain: {examples}")

    write_json(ann_out / "object_ann.json", normalized_object_ann)
    write_json(ann_out / "category.json", category)
    write_json(ann_out / "attribute.json", attribute)
    write_json(ann_out / "instance.json", instances)
    if traffic_light:
        write_json(ann_out / "traffic_light.json", traffic_light)
    else:
        path = ann_out / "traffic_light.json"
        if path.exists() or path.is_symlink():
            path.unlink()
    deprecated = ann_out / "traffic_light_map_association.json"
    if deprecated.exists() or deprecated.is_symlink():
        deprecated.unlink()

    print(f"derived dataset: {out}")
    print(
        f"object_ann: {len(normalized_object_ann)} | instances: {len(instances)} "
        f"| traffic_light relations: {len(traffic_light)} "
        f"({'wrote traffic_light.json' if traffic_light else 'traffic_light.json absent -> Tier B'})"
    )
    print(
        f"deprecated map associations consumed: {len(association_by_object_ann)} "
        f"| generated missing instances: {generated_missing_instances} "
        f"| split object_ann rows: {split_rows}"
    )


if __name__ == "__main__":
    main()
