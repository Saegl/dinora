import json
import math

import chess
import numpy as np
from tensorflow import keras

from policy import policy_from_move, flip_policy
from board_utils import canon_input_planes, is_black_turn
from preprocess_pgn import load_games, preprocess_games


def gen_planes(buffer):
    for pos in buffer:
        plane = canon_input_planes(pos['fen'])
        yield plane


def gen_results(buffer):
    for pos in buffer:
        res = pos['result']
        if is_black_turn(pos['fen']):
            res = -res
        yield pos['result']


def gen_policies(buffer):
    for pos in buffer:
        move = chess.Move.from_uci(pos['move'])
        policy = policy_from_move(move)
        if is_black_turn(pos['fen']):
            policy = flip_policy(policy)
        yield policy


class RawBufferDataset(keras.utils.Sequence):
    """Used for keras model.fit"""
    def __init__(self, buffer, batch_size: int):
        self.buffer = buffer
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.buffer) / self.batch_size)

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        buffer = self.buffer[start_index:end_index]

        planes = np.array(list(gen_planes(buffer)), dtype=np.float)
        results = np.array(list(gen_results(buffer)), dtype=np.float)
        policies = np.array(list(gen_policies(buffer)), dtype=np.float)

        batch_x = planes
        batch_y = [policies, results]
        return batch_x, batch_y


class GamesDataset(RawBufferDataset):
    """
    Load dataset from preprocessed games JSON file
    Example of JSON games can be found at /feed
    """

    def __init__(self, filename_buffer: str, batch_size: int):
        with open(filename_buffer, 'r', encoding='utf8') as f:
            buffer = json.load(f)
        super().__init__(buffer, batch_size)


class PGNDataset(RawBufferDataset):
    """
    Direct chess games loading from pgn file,
    without using JSON preprocessing file
    """

    def __init__(self, pgn_filename, batch_size: int, max_games: int):
        games = load_games(pgn_filename, max_games)
        buffer = preprocess_games(games)
        super().__init__(buffer, batch_size)
