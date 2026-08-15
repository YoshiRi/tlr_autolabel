#!/usr/bin/env python3
"""CLI wrapper for tlr_autolabel.cli.compare (GT-free comparison, L6)."""

import _bootstrap  # noqa: F401

from tlr_autolabel.cli.compare import compare_naive_main


if __name__ == "__main__":
    compare_naive_main()
