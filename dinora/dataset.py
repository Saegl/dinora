import tensorflow as tf

from .preprocess_pgn import load_chess_games, chess_positions
from .selfplay import gen_game

default_dinora_signature = (
    tf.TensorSpec(shape=(18, 8, 8), dtype=tf.float32, name=None),  # Board
    (
        tf.TensorSpec(shape=(1968,), dtype=tf.float32, name=None),  # Policy
        tf.TensorSpec(shape=(), dtype=tf.float32, name=None),  # Game result
    ),
)


def create_dataset_from_pgn(pgn_path: str, max_games: int) -> tf.data.Dataset:
    games = load_chess_games(pgn_path, max_games)
    return create_dataset_from_games(games)


def create_dataset_from_selfplay(nodes, net, c) -> tf.data.Dataset:
    games = [gen_game(nodes, net, c)]
    return create_dataset_from_games(games)


def create_dataset_from_games(games) -> tf.data.Dataset:
    tfdataset = tf.data.Dataset.from_generator(
        lambda: chess_positions(games), output_signature=default_dinora_signature
    )
    return tfdataset
