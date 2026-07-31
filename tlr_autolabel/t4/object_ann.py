"""Tier B object_ann/instance construction (REFACTOR_PLAN.md phase 4).

Greedy IoU 2D-instance tracking shared by to_object_ann.py. Instance
identity is a 2D image concept; map ids are consulted only to avoid
merging two annotations that would point one instance at multiple
traffic-light map primitives in traffic_light.json.
"""
import hashlib


def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def instance_token_for(ch, record_index, record):
    uid = record.get("uid") or f"record-{record_index}"
    box = ",".join(f"{float(v):.3f}" for v in record["box"])
    return hashlib.md5(f"instance|{ch}|{record['sd']}|{uid}|{box}".encode()).hexdigest()


def map_ids_compatible(track_map_id, record_map_id):
    return not track_map_id or not record_map_id or str(track_map_id) == str(record_map_id)


def merged_map_id(track_map_id, record_map_id):
    return str(track_map_id or record_map_id or "")


def assign_instances(records, sd_meta):
    """Greedy IoU tracking per camera channel across time -> instance_token per
    record. Instance identity is a 2D annotation concept; map ids are used only
    to avoid writing one instance -> multiple traffic-light linestring relations.
    Returns {record_idx: instance_token} and the instance table."""
    by_channel = {}
    for i, r in enumerate(records):
        ch, ts = sd_meta.get(r["sd"], ("", 0))
        by_channel.setdefault(ch, []).append((ts, i))
    inst_of = {}
    instances = []
    for ch, items in by_channel.items():
        items.sort()
        # (instance_token, last_box, first_ann_uid, category, last_ts, map_id)
        # The instance remains a 2D image identity, but a single instance cannot
        # point to multiple map linestrings in traffic_light.json. Preventing
        # incompatible map merges here keeps the relation table unambiguous.
        active = []
        last_ts = None
        for ts, i in items:
            r = records[i]
            if last_ts is not None and ts != last_ts:
                # new frame: keep only tracks touched last frame (active reset per frame)
                active = [a for a in active if a[4] == last_ts]
            best, bi = 0.3, -1
            for k, a in enumerate(active):
                if not map_ids_compatible(a[5], r.get("map_id")):
                    continue
                v = box_iou(r["box"], a[1])
                if v > best:
                    best, bi = v, k
            if bi >= 0:
                tok = active[bi][0]
                active[bi] = (tok, r["box"], active[bi][2], r["cat"], ts,
                              merged_map_id(active[bi][5], r.get("map_id")))
            else:
                tok = instance_token_for(ch, i, r)
                active.append((tok, r["box"], r["uid"], r["cat"], ts,
                               str(r.get("map_id") or "")))
                instances.append({"token": tok, "category_token": None,
                                  "instance_name": None, "_cat": r["cat"],
                                  "nbr_annotations": 0,
                                  "first_annotation_token": "", "last_annotation_token": ""})
            inst_of[i] = tok
            last_ts = ts
    return inst_of, instances
