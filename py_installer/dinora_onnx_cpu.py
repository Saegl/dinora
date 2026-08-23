from dinora.uci.cli import DefaultArgs, main

main(DefaultArgs(searcher="mcts", model="onnx", device="cpu"))
