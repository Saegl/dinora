"""
`dinora.uci.cli` has to be importable without loading any native runtime, so
that `--limit-threads` can set its env vars after argparse (see
`dinora.threads.limit_threads`). Keeping the registries lazy is what buys this,
it is easy to break by adding a convenience import somewhere on the way.
"""

import subprocess
import sys

HEAVY_MODULES = ["numpy", "torch", "onnxruntime"]


def test_cli_import_does_not_load_native_runtimes() -> None:
    source = f"""
import sys

import dinora.uci.cli

loaded = [m for m in {HEAVY_MODULES} if m in sys.modules]
if loaded:
    raise SystemExit("dinora.uci.cli imported: " + ", ".join(loaded))
"""

    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )

    assert proc.returncode == 0, proc.stderr
