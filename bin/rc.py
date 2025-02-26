# ruff: noqa
"""
Put this into PYTHONSTARTUP if you tinker in repl a lot

```bash
export PYTHONSTARTUP=bin/rc.py
```

Or run with `python -i bin/rc.py`
"""

import pathlib
import sys
from pathlib import Path

import chess
import numpy as np
import torch
from chess import Board

from dataset.encoders.compact_board_tensor import (
    board_to_compact_state,
    compact_state_to_board_tensor,
)
from dataset.encoders.outcome import z_value
from dinora.encoders.board_tensor import board_to_tensor
from dinora.encoders.policy import index_to_move, policy_index
from dinora.models import model_selector

model = model_selector("alphanet", Path("models/alphanet_mini.ckpt"), device="cuda")

fens = [
    "r3k1n1/5p2/p1p1p3/3pb3/7q/2N5/PPPP1PP1/R1B2RK1 w q - 4 17",
    "r4rk1/pbq1nppp/1p6/2pP2N1/3P4/2P1n1P1/PPQNB1K1/R1B3R1 w - - 7 19",
    "r4rk1/3b1ppp/pq2pn2/8/Ppp5/2N1PQ2/1P2B1PP/R4RK1 w - - 0 18",
    "rnbq1rk1/ppp2ppp/3b1n2/8/8/3B1N2/PPP2PPP/R1BQ1RK1 w - - 0 11",
    "r3r1k1/5ppp/1qp5/p7/2nPp2Q/2P5/P4RPP/4R1K1 w - - 0 24",
    "r3kb1r/pp6/5p1p/q2np3/1n1pN1bP/P4NB1/1PPQP1P1/2KR1B1R w kq - 1 16",
    "r4rk1/pp1b3p/4ppp1/3pP3/3qP3/7R/PPQ3PP/R1B3K1 w - - 0 19",
    "r1bqkb1r/5ppp/p3p3/1p1n4/2p2B2/P1N2Q2/1P3PPP/R3KB1R w KQkq - 1 13",
    "r5k1/p1p1qr1p/1p1pNnpR/8/3P4/2PBP3/PP4P1/2K4Q w - - 1 23",
    "r1b3kr/ppp5/2np4/2bNp1qp/4P1pN/3P2PP/PPP3P1/R2QKR2 w Q - 4 15",
    "2r2rk1/pp2bppp/2n1p3/3q4/2NP4/4B3/PP2QPPP/R2R2K1 w - - 0 15",
    "r2qkb1r/pp3ppp/2n1pn2/2Pp4/3P4/5P1P/PP3P2/RNBQKB1R w KQkq - 1 9",
    "r3r1k1/ppqn1ppp/8/2p2p2/1BPPp3/2P1PP2/P1Q3PP/R4RK1 w - - 0 17",
    "r3r1k1/ppq2ppp/2p5/8/1BPPp1n1/2P1P3/P4QPP/R4RK1 w - - 2 19",
    "rn1qr1k1/pbp1bp1p/1p1pp1pB/8/3P2n1/2PBPN2/PPQN1PPP/2KR3R w - - 4 11",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqk2r/ppppbppp/4pn2/8/8/5NP1/PPPPPPBP/RNBQK2R w KQkq - 3 4",
    "1R6/8/8/5rkp/6p1/8/5PK1/8 w - - 12 54",
    "r1b1k2r/ppppqppp/2n5/4n3/2P5/2P1PN2/P3BPPP/R1BQK2R w KQkq - 0 9",
    "r2q1rk1/ppp2pp1/2nb1n1p/7b/2BP4/2N1BN1P/PP3PP1/R2Q1RK1 w - - 1 12",
    "rnbqkbnr/p2ppppp/8/2pP4/1p2P3/8/PPP2PPP/RNBQKBNR w KQkq - 0 4",
    "rn1qkbnr/1bpppp2/pp4p1/7p/2PPP2P/2N5/PP3PP1/R1BQKBNR w KQkq - 0 6",
    "8/pp2nkp1/8/2P1R3/8/2B4P/P1P2PK1/3r4 w - - 5 35",
    "r3k2r/4bppp/p1p1pn2/q1Pp4/3P4/2N2P1P/PP1B1P2/R2QK2R w KQkq - 2 13",
    "r2q1rk1/ppp2pp1/2np1n1p/4p3/3bP2B/1PNP1N1P/1PP2PP1/R2QK2R w KQ - 1 11",
    "rn2k2r/1bpqn1b1/pp1p1pp1/3Pp1Bp/1PP1P2P/2NB1N2/P4PP1/R2QR1K1 w kq - 0 13",
    "r4rk1/p1p1qppp/1pnpb3/8/P1nN1P2/B1P1P3/4B1PP/R2Q1RK1 w - - 0 14",
]

boards = [Board(fen=fen) for fen in fens]

print("rc.py loaded")
