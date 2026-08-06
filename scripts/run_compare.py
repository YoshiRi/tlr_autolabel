#!/usr/bin/env python3
"""CLI wrapper for tlr_autolabel.cli.compare (matrix runner, L1)."""

import _bootstrap  # noqa: F401

from tlr_autolabel.cli.compare import run_compare_main


if __name__ == "__main__":
    run_compare_main()
