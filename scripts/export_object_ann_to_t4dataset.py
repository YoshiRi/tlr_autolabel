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

CLI wrapper only: the normalization logic lives in tlr_autolabel/t4/adapters.py
(REFACTOR_PLAN.md phase 4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from tlr_autolabel.t4.adapters import (
    association_map,
    build_traffic_light,
    load_json,
    normalize_object_ann,
    rebuild_instances,
    resolve_path,
    sample_data_timestamps,
    scene_name,
    symlink_dataset,
    write_json,
)


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
    stale_instance_map = ann_out / "traffic_light_instance_map.json"
    if stale_instance_map.exists() or stale_instance_map.is_symlink():
        stale_instance_map.unlink()
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
