import logging
import chess
import chess.pgn
import json

logging.basicConfig(level=logging.DEBUG)


def num_result(game):
    result = game.headers['Result']
    if result == '0-1':
        result = 0.0
    elif result == '1-0':
        result = 1.0
    elif result == '1/2-1/2':
        result = 0.5
    else:
        raise ValueError(f"Illegal game result: {result}")
    return result


def load_games(filename_pgn: str):
    pgn = open(filename_pgn)
    game = True
    i = 0
    while game:
        game = chess.pgn.read_game(pgn)
        if not game or game.headers["Variant"] != "Standard":
            game = chess.pgn.read_game(pgn)
            continue
        i += 1
        if i % 250 == 0:
            logging.info(f'{i} games loaded')
        yield game


def preprocess_games(games):
    buffer = []
    for game in games:
        result = num_result(game)
        board = game.board()

        for move in game.mainline_moves():
            fen = board.fen()
            buffer.append({
                'fen': fen,
                'result': result,
                'move': move.uci(),
            })

            try:
                board.push(move)
            except AssertionError:
                logging.warning("Broken game found,"
                    f"can't make a move {move}."
                    "Skipping")
    return buffer


def dump_games(filename_buffer: str, buffer):
    logging.debug("Games dumping started")
    with open(filename_buffer, 'w', encoding='utf8') as f:
        json.dump(buffer, f)
    logging.debug("Dump success")


def main(filename_pgn: str, filename_buffer: str):
    games = load_games(filename_pgn)
    buffer = preprocess_games(games)
    dump_games(filename_buffer, buffer)


if __name__ == '__main__':
    filename_pgn = "pgn/All_My_Games.pgn"
    filename_buffer = "feed/games.json"
    main(filename_pgn, filename_buffer)
