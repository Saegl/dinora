import tensorflow as tf

from .preprocess_pgn import load_chess_games, chess_positions


def create_dataset_from_pgn(pgn_path: str, max_games: int) -> tf.data.Dataset:
    games = load_chess_games(pgn_path, max_games)
    tfdataset = tf.data.Dataset.from_generator(
        lambda: chess_positions(games),
        output_signature=(
            tf.TensorSpec(shape=(18, 8, 8), dtype=tf.float32, name=None),
            (
                tf.TensorSpec(shape=(1968,), dtype=tf.float32, name=None),
                tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
            )
        )
    )
    return tfdataset
