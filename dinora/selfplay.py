import chess
from chess.pgn import Game

from .utils import disable_tensorflow_log

disable_tensorflow_log()
from .search import uct_nodes


def gen_game(nodes: int, net, c_puct, dirichlet_alpha, noise_eps, softmax_temp):
    game = Game()
    node = game
    while not node.board().is_game_over(claim_draw=True):
        move, _ = uct_nodes(
            node.board(),
            nodes,
            net,
            c_puct,
            None,
            dirichlet_alpha,
            noise_eps,
            softmax_temp,
        )
        move = chess.Move.from_uci(move)
        node = node.add_variation(move)
        print(move, end=" ", flush=True)
    print(node.board().result(claim_draw=True))

    result = node.board().result(claim_draw=True)
    game.headers["Result"] = result
    return game


if __name__ == "__main__":
    from .net import ChessModel

    nodes = 10
    net = ChessModel("models/best_light_model.h5")
    c = 2.0
    game = gen_game(nodes, net, c)
    print(game)
