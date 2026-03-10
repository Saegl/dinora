import os
import sys

# TODO: We cannot just set `limit_threads` after `cli` import in `argparse`
# because numpy going to be imported
# cli.py -> engine.py -> search/registered.py -> MCTS -> noise.py -> import numpy as np
# requires major refactor

if "--limit-threads" in sys.argv:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

from dinora.uci.cli import DefaultArgs, main

main(DefaultArgs(searcher="mcts", model="onnx", device="cpu"))
