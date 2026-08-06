"""Detector/classifier comparison: run a matrix of configurations (L1) and
compare their raw outputs without GT or a map (L6).

See docs/inference_comparison.md. `matrix` runs inference; `naive` and `grid`
only read Tier A, so the layer separation the repo relies on ("no layer after L1
re-runs inference") holds inside the harness too.
"""
