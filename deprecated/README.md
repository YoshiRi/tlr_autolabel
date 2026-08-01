# deprecated tools

Superseded by the standard-t4 pivot (2026-07-23, see the repo README "Architecture").
Kept for reference / one-off use; not part of the maintained pipeline.

- `export_awml.py` — built an AWML-style derived dataset (object_ann with db_tlr
  categories). **Replaced by `../scripts/to_object_ann.py`** (L2 A→B), which writes a
  strictly-standard `object_ann.json`; AWML `create_data_t4dataset.py` then reads
  it directly. No custom AWML view needed.
- `export_labels.py` — wrote COCO / CVAT-1.1 from our v1 sidecar. **Replaced by**
  producing standard `object_ann.json` (`../scripts/to_object_ann.py`) and letting the
  existing t4devkit / webauto tooling convert to COCO / CVAT / Deepen.

The canonical→db_tlr vocabulary now lives in `../configs/state_vocab/db_tlr.yaml`
and is applied by `../scripts/to_object_ann.py`.
