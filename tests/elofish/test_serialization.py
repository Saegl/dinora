import json
import pathlib

from elofish.elofish import MatchConfig


def test_decoder_encoder():
    path = pathlib.Path("configs/elofish/stockfish_dinora_mcts_batch.json")

    with path.open("r") as f:
        original_dict = json.load(f)

    config = MatchConfig.from_file(path)
    encoded_dict = config.to_dict()

    assert original_dict == encoded_dict
