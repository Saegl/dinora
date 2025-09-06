import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dinora.uci.cli import DefaultArgs, main

main(DefaultArgs(searcher="mcts", model="onnx", device="cpu"))
