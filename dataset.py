import json
import math

import chess
import numpy as np
from tensorflow import keras

from policy import policy_from_move, flip_policy
from board_utils import canon_input_planes, is_black_turn


def gen_planes(buffer):
    for pos in buffer:
        plane = canon_input_planes(pos['fen'])
        yield plane


def gen_results(buffer):
    for pos in buffer:
        yield pos['result']


def gen_policies(buffer):
    for pos in buffer:
        move = chess.Move.from_uci(pos['move'])
        policy = policy_from_move(move)
        if is_black_turn(pos['fen']):
            policy = flip_policy(policy)
        yield policy


class GamesDataset(keras.utils.Sequence):
    def __init__(self, filename_buffer, batch_size):
        with open(filename_buffer, 'r', encoding='utf8') as f:
            self.buffer = json.load(f)
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
