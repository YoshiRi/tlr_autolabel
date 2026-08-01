"""TensorRT .engine inference backend (REFACTOR_PLAN.md phase 6).

Extracted from tlr_autolabel.py.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# repo root: tlr_autolabel/inference/trt.py -> tlr_autolabel/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
TRT_RUN_SOURCE = REPO_ROOT / "tools" / "trt_run.cpp"
TRT_RUN_BINARY = REPO_ROOT / "build" / "trt_run"


class TrtServer:
    """Runs a TensorRT .engine through the trt_run helper (serve mode keeps the
    engine deserialized across images). Single input / single output engines
    only. Compiled from tools/trt_run.cpp on first use."""

    def __init__(self, engine_path):
        binary = str(TRT_RUN_BINARY)
        if not os.path.exists(binary):
            TRT_RUN_BINARY.parent.mkdir(parents=True, exist_ok=True)
            cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            subprocess.check_call(
                ["g++", "-O2", str(TRT_RUN_SOURCE), "-o", binary,
                 f"-I{cuda}/include", f"-L{cuda}/lib64",
                 "-lnvinfer", "-lcudart"])
        self.proc = subprocess.Popen([binary, engine_path, "serve"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True)
        self.in_shape = self.out_shape = None
        for line in self.proc.stdout:
            t = line.split()
            if t and t[0] == "INPUT":
                self.in_shape = [int(x) for x in t[1:]]
            elif t and t[0] == "OUTPUT":
                self.out_shape = [int(x) for x in t[1:]]
            elif t and t[0] == "READY":
                break
        assert self.in_shape and self.out_shape, "trt_run did not report tensor shapes"
        self.tmp = tempfile.mkdtemp(prefix="trt_run_")

    def run(self, blob):
        ip = os.path.join(self.tmp, "in.f32")
        op = os.path.join(self.tmp, "out.f32")
        blob.astype(np.float32).tofile(ip)
        self.proc.stdin.write(f"{ip} {op}\n")
        self.proc.stdin.flush()
        resp = self.proc.stdout.readline().strip()
        if resp != "DONE":
            raise RuntimeError(f"trt_run inference failed: {resp!r}")
        return np.fromfile(op, dtype=np.float32).reshape(self.out_shape)
