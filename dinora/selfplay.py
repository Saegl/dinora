"""
Selfplay is not currently supported
and this module is absolutely broken
"""
# pragma: no cover
import chess
from chess.pgn import Game, GameNode

from .utils import disable_tensorflow_log

disable_tensorflow_log()
from .search import uct_nodes


def gen_game(
    nodes: int, net, c_puct, dirichlet_alpha, noise_eps, softmax_temp
):  # pragma: no cover
    game = Game()
    node: GameNode = game
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
        node = node.add_variation(move)
        print(move, end=" ", flush=True)
    print(node.board().result(claim_draw=True))

    result = node.board().result(claim_draw=True)
    game.headers["Result"] = result
    return game
