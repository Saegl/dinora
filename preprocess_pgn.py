import json
import logging

import chess
import chess.pgn

logging.basicConfig(level=logging.DEBUG)


def num_result(game):
    result = game.headers['Result']
    if result == '0-1':
        result = -1.0
    elif result == '1-0':
        result = 1.0
    elif result == '1/2-1/2':
        result = 0.0
    else:
        raise ValueError(f"Illegal game result: {result}")
    return result


def load_games(filename_pgn: str, max_games: int = 100000):
    pgn = open(filename_pgn, 'r', encoding='utf8', errors='ignore')
    game = True
    i = 0
    while game:
        game = chess.pgn.read_game(pgn)
        if not game or game.headers.get("Variant", "Standard") != "Standard":
            game = chess.pgn.read_game(pgn)
            continue
        i += 1
        if i > max_games:
            break
        if i % 150 == 0:
            logging.info(f'{i}/{max_games}')
        yield game


def positions(games):
    for game in games:
        try:
            result = num_result(game)
        except ValueError:
            continue
        board = game.board()

        for move in game.mainline_moves():
            fen = board.fen()
            yield {
                'fen': fen,
                'result': result,
                'move': move.uci(),
            }

            try:
                board.push(move)
            except AssertionError:
                logging.warning("Broken game found,"
                                f"can't make a move {move}."
                                "Skipping")


def preprocess_games(games):
    buffer = list(positions(games))
    return buffer


def dump_games(filename_buffer: str, buffer):
    logging.debug("Games dumping started")
    with open(filename_buffer, 'w', encoding='utf8') as f:
        json.dump(buffer, f)
    logging.debug("Dump success")


def main(filename_pgn: str, filename_buffer: str, max_games: int = 100000):
    games = load_games(filename_pgn, max_games)
    buffer = preprocess_games(games)
    dump_games(filename_buffer, buffer)


if __name__ == '__main__':
    filename_pgn = "pgn/lichess_elite_2021-06.pgn"
    filename_buffer = "feed/games.json"
    main(filename_pgn, filename_buffer)
