from itertools import product
from statistics import mean

import chess
import chess.engine
from chess.pgn import Game

from .utils import disable_tensorflow_log

disable_tensorflow_log()
from .search import uct_nodes
from .net import ChessModel

output_filename = "games.pgn"
number_of_games = 2
elo_k = 32

stockfish_elo = [1350]  # [1400, 1600, 1800]
stockfish_nodes = 100_000

dinora_nodes = 50
dinora_c = 2.0
dinora_net = ChessModel("models/best_light_model.h5")
dinora_elo = mean(stockfish_elo)
dinora_name = "Dinora"

fish = chess.engine.SimpleEngine.popen_uci("stockfish")
fish.configure({"UCI_LimitStrength": True, "UCI_Elo": 1400})
output_file = open(output_filename, "w", encoding="ascii")

color = True
for fish_elo, game_n in product(stockfish_elo, range(number_of_games)):
    stockfish_name = f"<Stockfish {fish_elo} nodes {stockfish_nodes}>"
    fish.configure(
        {
            "UCI_Elo": fish_elo,
        }
    )
    dinora_color = color
    if dinora_color:
        white = dinora_name
        black = stockfish_name
    else:
        white = stockfish_name
        black = dinora_name

    print(f"Dinora Elo: {dinora_elo}")
    print(f"{white} vs {black}")

    game = Game()
    game.headers["Event"] = "Dinora Elo Estimating"
    game.headers["White"] = white
    game.headers["Black"] = black

    node = game
    while not node.board().is_game_over(claim_draw=True):
        if dinora_color:
            # move = fish.play(node.board(), limit=chess.engine.Limit(nodes=100_000)).move
            move, _ = uct_nodes(node.board(), dinora_nodes, dinora_net, dinora_c)
            move = chess.Move.from_uci(move)
        else:
            move = fish.play(node.board(), limit=chess.engine.Limit(nodes=100_000)).move
        node = node.add_variation(move)
        dinora_color = not dinora_color
        print(".", end="", flush=True)
    print()  # newline after dots

    result = node.board().result(claim_draw=True)
    game.headers["Result"] = result
    print(game, file=output_file, end="\n\n")

    # Update Dinora Elo rating
    dinora_estimated = 1 / (1 + 10 ** ((fish_elo - dinora_elo) / 400))
    if result == "*":
        raise ValueError("Game is broken")
    elif (result == "1-0" and color) or (result == "0-1" and not color):
        score = 1.0
    elif result == "1/2-1/2":
        score = 0.5
    else:
        score = 0.0
    dinora_elo = dinora_elo + elo_k * (score - dinora_estimated)

    print(f"Dinora game score: {score}")
    print(f"Dinora new Elo: {dinora_elo}\n")
    color = not color

print(f"Final Dinora Elo: {dinora_elo}")
fish.quit()
output_file.close()
