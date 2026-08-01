"""Compatibility shim: moved to tlr_autolabel/tracking/temporal.py.

See REFACTOR_PLAN.md step 3 (Move Pure Core Modules First). New code should
import from tlr_autolabel.tracking.temporal; this shim keeps the old
top-level import path working until all scripts are migrated.
"""
from tlr_autolabel.tracking.temporal import (  # noqa: F401
    BBox,
    LowDetectionMatch,
    ObservedTrack,
    PropagatedTrack,
    TemporalAssociator,
    TemporalTrack,
    TemporalTrackingConfig,
    TrackingResult,
    center,
    center_distance,
    diag_size_ratio,
    diagonal,
    iou,
    min_side,
)
