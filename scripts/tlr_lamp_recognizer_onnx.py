#!/usr/bin/env python3
"""Debug CLI for the shared LampRecognizer ONNX decoder."""

import _bootstrap  # noqa: F401

from tlr_autolabel.inference.lamp_recognizer_onnx import main


if __name__ == "__main__":
    main()
