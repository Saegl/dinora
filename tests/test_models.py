import pytest
import chess

from dinora.models.dnn import DNNModel
from dinora.models.badgyal import BadgyalModel


@pytest.mark.parametrize("model", [DNNModel(), BadgyalModel()])
def test_white_winning_white_to_move(model):
    fen = "4qn1k/2R3pp/p4p2/3Q4/P4N2/6P1/5PBP/2R3K1 w - - 0 31"
    board = chess.Board(fen)
    _, value_estimate = model.raw_eval(board)
    assert value_estimate >= 0.5


@pytest.mark.parametrize("model", [DNNModel(), BadgyalModel()])
def test_white_winning_black_to_move(model):
    fen = "4qn1k/p1R3pp/5p2/3Q4/P4N2/6P1/5PBP/2R3K1 b - - 0 30"
    board = chess.Board(fen)
    _, value_estimate = model.raw_eval(board)
    assert value_estimate <= -0.5


@pytest.mark.parametrize("model", [DNNModel(), BadgyalModel()])
def test_black_winning_white_to_move(model):
    fen = "r1r1q2k/5bpp/5p2/p3n3/P7/6P1/3Q1P1P/2R3K1 w - - 0 31"
    board = chess.Board(fen)
    _, value_estimate = model.raw_eval(board)
    assert value_estimate <= -0.5


@pytest.mark.parametrize("model", [DNNModel(), BadgyalModel()])
def test_black_winning_black_to_move(model):
    fen = "r1r1q2k/p4bpp/5p2/4n3/P7/6P1/3Q1P1P/2R3K1 b - - 0 30"
    board = chess.Board(fen)
    _, value_estimate = model.raw_eval(board)
    assert value_estimate >= 0.5
