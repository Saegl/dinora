import chess
import chess.engine


def test_uci():
    uci = chess.engine.SimpleEngine.popen_uci(
        ["python", "-m", "dinora", "--model=onnx", "--device=cpu"]
    )

    board = chess.Board()
    play_res = uci.play(board, chess.engine.Limit(time=0.1))

    assert play_res.move is not None
    assert play_res.move.uci() in ["e2e4", "d2d4", "c2c4", "Nf3"]

    uci.close()
