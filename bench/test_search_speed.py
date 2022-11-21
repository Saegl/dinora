import chess
from dinora.mcts import run_mcts, NodesCountConstraint, MCTSparams
from dinora.models.dnn import DNNModel
from dinora.models.cached_model import CachedModel

# model = DNNModel()
model = CachedModel(DNNModel())


def search_position():
    board = chess.Board()
    run_mcts(board, NodesCountConstraint(200), model, MCTSparams())

    print(model.hit_ratio)
    model.clear()


def search_game():
    board = chess.Board()
    for _ in range(10):
        node = run_mcts(board, NodesCountConstraint(200), model, MCTSparams())
        move = node.get_most_visited_node().move
        board.push(move)

    # print(model.hit_ratio)
    # print(board.fen)
    # model.clear()


def test_search_position_speed(benchmark):
    benchmark(search_position)


# pytest tests/bench/test_search_speed.py::test_search_game_speed
#  --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-min-rounds=2
def test_search_game_speed(benchmark):
    benchmark(search_game)


if __name__ == "__main__":
    search_game()
