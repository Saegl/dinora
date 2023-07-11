import chess
import chess.engine
from dinora.handmade_val_dataset.dataset import POSITIONS


engine = chess.engine.SimpleEngine.popen_uci('stockfish')


for pos in POSITIONS:
    board = chess.Board(fen=pos['fen'])
    text = pos['text']

    info = engine.analyse(board, limit=chess.engine.Limit(nodes=100_000))
    score = info['score']
    pos['stockfish_cp'] = score.white().score(mate_score=10_000) / 100
    wdl = score.wdl(ply=board.ply()).pov(board.turn)
    win = wdl.winning_chance()
    draw = wdl.drawing_chance()
    loss = wdl.losing_chance()
    pos['stockfish_wdl'] = f"w {win:.2f} | d {draw:.2f} | l {loss:.2f}"


engine.quit()

import pprint
pprint.pprint(POSITIONS)
