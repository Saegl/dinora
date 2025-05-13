import argparse
import json
import pathlib
from collections.abc import Callable
from typing import Any

import chess
import numpy as np
import numpy.typing as npt
import torch

from dataset.encoders.compact_board_tensor import compact_state_to_board_tensor
from dinora.encoders.policy import extract_logit
from dinora.engine import Engine
from dinora.models import model_selector
from dinora.models.alphanet import AlphaNet
from dinora.search.stoppers import MoveTime, NodesCount, Stopper

device = "cuda"

npuint64 = npt.NDArray[np.uint64]
npf32 = npt.NDArray[np.float32]
npf64 = npt.NDArray[np.float64]

StopperCreator = Callable[[], Stopper]


def calc_policy_cploss(
    model: AlphaNet,
    positions: list[dict[Any, Any]],
    policy_boards: npuint64,
    batch_size: int,
) -> npf64:
    total_positions = len(positions)
    move_losses = []

    for batch_start in range(0, total_positions, batch_size):
        batch_end = batch_start + batch_size
        batch_positions = positions[batch_start:batch_end]
        batch_boards = policy_boards[batch_start:batch_end]

        boards_tensor = torch.from_numpy(
            np.array([compact_state_to_board_tensor(b) for b in batch_boards])
        ).to(device)

        with torch.no_grad():
            policy, _ = model(boards_tensor)

        for i, position in enumerate(batch_positions):
            flip = position["flip"]
            uci_moves = list(position["actions"])

            move_logits = np.array(
                [
                    extract_logit(policy[i], chess.Move.from_uci(move), flip)
                    for move in uci_moves
                ]
            )
            best_move_index = move_logits.argmax()
            best_move = uci_moves[best_move_index]
            move_loss = position["actions"][best_move]
            move_losses.append(move_loss)

    return np.array(move_losses)


def calc_value_cploss(
    model: AlphaNet,
    positions: list[dict[Any, Any]],
    value_boards: npuint64,
    batch_size: int,
) -> npf64:
    npf32 = npt.NDArray[np.float32]

    def load_batch(batch_offset: int) -> npf32:
        value_boards_np = np.array(
            [
                compact_state_to_board_tensor(b)
                for b in value_boards[batch_offset : batch_offset + batch_size]
            ]
        )
        boards = torch.from_numpy(value_boards_np).to(device)

        with torch.no_grad():
            _, raw_value = model(boards)

        value_count = len(raw_value)
        value: npf32 = raw_value.cpu().numpy().reshape(value_count)
        return value

    batch_offset = 0
    value = load_batch(batch_offset)

    def get_value_positions(pos_start: int, pos_end: int) -> npf32:
        nonlocal batch_offset, value
        if pos_start > pos_end:
            return np.array([])

        batch_pos_start = pos_start - batch_offset
        batch_pos_end = min(batch_size, pos_end - batch_offset)

        prefix = value[batch_pos_start:batch_pos_end]

        if pos_end - batch_offset <= batch_size:
            return prefix

        batch_offset += batch_size
        value = load_batch(batch_offset)

        return np.concatenate([prefix, get_value_positions(batch_offset, pos_end)])

    move_losses = []
    moves_offset = 0
    for i in range(len(positions)):
        position = positions[i]
        moves_uci_seq = position["moves_uci_seq"]

        pos_start = moves_offset
        pos_end = moves_offset + len(moves_uci_seq)

        values = -get_value_positions(pos_start, pos_end)

        best_index = values.argmax()
        best_move = moves_uci_seq[best_index]

        assert len(moves_uci_seq) == len(values)

        move_loss = position["actions"][best_move]
        move_losses.append(move_loss)

        moves_offset += len(moves_uci_seq)

    return np.array(move_losses)


def calc_engine_cploss(
    model: AlphaNet,
    positions: list[dict[Any, Any]],
    searcher: str,
    stopper_creator: StopperCreator,
    params: dict[str, Any],
) -> npf64:
    engine = Engine(searcher=searcher)
    engine._model = model
    engine.searcher.update_from_dict(params)

    move_losses = []
    for position in positions:
        board = chess.Board(fen=position["fen"])
        move = engine.get_best_move(board, stopper=stopper_creator())
        move_loss = position["actions"][move.uci()]
        move_losses.append(move_loss)

    return np.array(move_losses)


def make_stopper_creator(
    movetime: float | None = None, nodes: int | None = None
) -> StopperCreator:
    if nodes is not None:
        return lambda: NodesCount(nodes)
    elif movetime is not None:
        return lambda: MoveTime(int(1000 * movetime))
    else:
        raise Exception("Cant create stopper")


def load_cploss(
    loaddir: pathlib.Path, count: int
) -> tuple[npuint64, npuint64, list[dict[Any, Any]]]:
    value_boards_file = loaddir / "value_boards.npz"
    boards_file = loaddir / "policy_boards.npz"
    positions_file = loaddir / "positions.json"

    value_boards = np.load(value_boards_file)["boards"]
    policy_boards = np.load(boards_file)["boards"]
    positions = json.load(positions_file.open())

    return value_boards, policy_boards, positions[:count]


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model_path")
    argparser.add_argument("batch_size")
    argparser.add_argument("loaddir")
    argparser.add_argument("--enable_engine", default=False)
    argparser.add_argument("--max_positions", default=99_999, type=int)
    argparser.add_argument("--searcher", default="auto")
    argparser.add_argument("--movetime", default=1.0, type=float)
    argparser.add_argument("--nodes", type=int)

    args = argparser.parse_args()

    model_path = pathlib.Path(args.model_path)
    batch_size = int(args.batch_size)
    max_positions = args.max_positions
    searcher = args.searcher
    movetime = args.movetime
    nodes = args.nodes
    params: dict[str, str] = {}  # TODO: pass custom params from somewhere?

    print("Model loading")
    model = model_selector("alphanet", model_path, "cuda")
    print("Model loaded")

    assert isinstance(model, AlphaNet)

    loaddir = pathlib.Path(args.loaddir)
    value_boards, policy_boards, positions = load_cploss(loaddir, max_positions)

    print(f"Positions: {len(positions)}")

    policy_cploss = calc_policy_cploss(model, positions, policy_boards, batch_size)
    policy_percentiles = np.percentile(policy_cploss, [10, 25, 50, 75, 90])
    print("Policy loss:")
    print(f"\tAverage: {np.mean(policy_cploss):.3f}")
    print(f"\tStd Dev: {np.std(policy_cploss):.3f}")
    print(f"\tMax: {np.max(policy_cploss):.3f}")
    print(f"\tTop 0cp: {np.mean(policy_cploss <= 0.0) * 100:.3f}%")
    print(f"\tTop 50cp: {np.mean(policy_cploss <= 50.0) * 100:.3f}%")
    print(f"\t10th Percentile: {policy_percentiles[0]:.3f}")
    print(f"\t25th Percentile: {policy_percentiles[1]:.3f}")
    print(f"\t50th Percentile: {policy_percentiles[2]:.3f}")
    print(f"\t75th Percentile: {policy_percentiles[3]:.3f}")
    print(f"\t90th Percentile: {policy_percentiles[4]:.3f}")
    print()

    value_cploss = calc_value_cploss(model, positions, value_boards, batch_size)
    val_percentiles = np.percentile(value_cploss, [10, 25, 50, 75, 90])
    print("Value loss:")
    print(f"\tAverage: {np.mean(value_cploss):.3f}")
    print(f"\tStd Dev: {np.std(value_cploss):.3f}")
    print(f"\tMax: {np.max(value_cploss):.3f}")
    print(f"\tTop 0cp: {np.mean(value_cploss <= 0.0) * 100:.3f}%")
    print(f"\tTop 50cp: {np.mean(value_cploss <= 50.0) * 100:.3f}%")
    print(f"\t10th Percentile: {val_percentiles[0]:.3f}")
    print(f"\t25th Percentile: {val_percentiles[1]:.3f}")
    print(f"\t50th Percentile: {val_percentiles[2]:.3f}")
    print(f"\t75th Percentile: {val_percentiles[3]:.3f}")
    print(f"\t90th Percentile: {val_percentiles[4]:.3f}")
    print()

    if args.enable_engine:
        stopper_creator = make_stopper_creator(movetime, nodes)
        print(f"Chosen Engine {searcher} {stopper_creator()}")
        engine_cploss = calc_engine_cploss(
            model, positions, searcher, stopper_creator, params
        )
        engine_percentiles = np.percentile(engine_cploss, [10, 25, 50, 75, 90])
        print("Engine loss:")
        print(f"\tAverage: {np.mean(engine_cploss):.3f}")
        print(f"\tStd Dev: {np.std(engine_cploss):.3f}")
        print(f"\tMax: {np.max(engine_cploss):.3f}")
        print(f"\tTop 0cp: {np.mean(engine_cploss <= 0.0) * 100:.3f}%")
        print(f"\tTop 50cp: {np.mean(engine_cploss <= 50.0) * 100:.3f}%")
        print(f"\t10th Percentile: {engine_percentiles[0]:.3f}")
        print(f"\t25th Percentile: {engine_percentiles[1]:.3f}")
        print(f"\t50th Percentile: {engine_percentiles[2]:.3f}")
        print(f"\t75th Percentile: {engine_percentiles[3]:.3f}")
        print(f"\t90th Percentile: {engine_percentiles[4]:.3f}")
        print()


if __name__ == "__main__":
    main()
