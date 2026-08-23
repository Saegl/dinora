import os

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
]


def limit_threads() -> None:
    """
    Limit numpy/onnxruntime/torch to a single thread.

    These variables are read when the underlying native runtime is loaded, which
    happens on `import numpy` / `import onnxruntime` / `import torch`. So this
    has to run before the first of those imports, but plain process start is
    early enough only by accident - calling it right after argparse works as
    long as nothing heavy is imported at module level on the way to `main`.

    `tests/test_import_weight.py` guards that property.
    """
    for var in THREAD_ENV_VARS:
        os.environ[var] = "1"
