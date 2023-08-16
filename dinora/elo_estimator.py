import datetime
import itertools
import random

import chess
import chess.pgn
import chess.engine

from dinora import glicko2
from dinora.glicko2 import Glicko2
from dinora.engine import Engine
from dinora.mcts.constraints import NodesCountConstraint


class RatedPlayer:
    name: str

    def init_rating(self, env: Glicko2):
        self.rating = env.create_rating()
    
    def close(self):
        pass


class UCIPlayer(RatedPlayer):
    def __init__(self, env: Glicko2, command: str):
        self.init_rating(env)
        self.uci_engine = chess.engine.SimpleEngine.popen_uci(command)
        self.uci_engine.configure({"UCI_LimitStrength": True})
        self.name = command

    def play(self, board: chess.Board) -> chess.Move:
        playres = stockfish_player.uci_engine.play(board, limit=chess.engine.Limit(nodes=10000))
        return playres.move

    def close(self):
        self.uci_engine.close()


class DinoraPlayer(RatedPlayer):
    def __init__(self, env: Glicko2) -> None:
        self.init_rating(env)
        self.engine = Engine("alphanet")
        self.engine.load_model()
        self.name = "dinora"
    
    def play(self, board: chess.Board) -> chess.Move:
        return self.engine.get_best_move(board, NodesCountConstraint(3))


MAX_GAMES_COUNT = 100

env = Glicko2()

dinora_player = DinoraPlayer(env)
stockfish_player = UCIPlayer(env, "stockfish")

############# CPU FIX
dinora_player.engine.mcts_params.send_func = lambda x: None
dinora_player.engine._model = dinora_player.engine._model.to('cpu')

game_ind = 0

while dinora_player.rating.phi > 75.0 and game_ind < MAX_GAMES_COUNT:
    stockfish_player.rating = env.create_rating(dinora_player.rating.mu)
    stockfish_player.uci_engine.configure({"UCI_Elo": min(3190, max(1320, int(stockfish_player.rating.mu)))})

    players = random.choice([[dinora_player, stockfish_player], [stockfish_player, dinora_player]])
    white_player = players[0]
    black_player = players[1]
    
    board = chess.Board()
    game = chess.pgn.Game(headers={
        'Event': 'Elo estimate',
        'Site': 'Dinora elo_estimator.py',
        'Stage': f"Dinora phi: {int(dinora_player.rating.phi)}",
        'Date': datetime.date.today().strftime(r"%Y.%m.%d"),
        'White': white_player.name,
        'Black': black_player.name,
        'Round': str(game_ind),
        'WhiteElo': str(int(white_player.rating.mu)),
        'BlackElo': str(int(black_player.rating.mu)),
    })
    node = game

    for player in itertools.cycle(players):
        if not board.outcome(claim_draw=True):
            move = player.play(board)
            node = node.add_variation(move)
            board.push(move)
        else:
            break


    result = board.result(claim_draw=True)
    game.headers['Result'] = result

    if result == "1-0" and white_player == dinora_player or result == "0-1" and black_player == dinora_player:
        dinora_outcome = glicko2.WIN
    elif result == "1/2-1/2":
        dinora_outcome = glicko2.DRAW
    else:
        dinora_outcome = glicko2.LOSS


    dinora_player.rating = env.rate(dinora_player.rating, [(dinora_outcome, stockfish_player.rating)])

    print(game, end="\n\n", flush=True)
    
    game_ind += 1


for player in players:
    player.close()
