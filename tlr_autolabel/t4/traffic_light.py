"""Tier B' `traffic_light.json` row construction (REFACTOR_PLAN.md phase 4).

Shared by to_object_ann.py and export_object_ann_to_t4dataset.py. Schema:
{token, instance_token, primitive_id} — see docs/existing_annotation_if.md.
"""
from tlr_autolabel.core.io import token_of


def build_traffic_light_row(instance_token: str, map_id: str) -> dict:
    return {
        "token": token_of("traffic_light", instance_token, map_id),
        "instance_token": instance_token,
        "primitive_id": map_id,
    }
