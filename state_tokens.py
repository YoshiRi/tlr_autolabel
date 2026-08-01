"""Compatibility shim: moved to tlr_autolabel/core/state_tokens.py.

See REFACTOR_PLAN.md step 3 (Move Pure Core Modules First). New code should
import from tlr_autolabel.core.state_tokens; this shim keeps the old
top-level import path working until all scripts are migrated.
"""
from tlr_autolabel.core.state_tokens import (  # noqa: F401
    CANON_RE,
    LEGACY_RE,
    MAP_BULB_COLOR,
    bulb_color,
    elements_key,
    parse_state,
)
