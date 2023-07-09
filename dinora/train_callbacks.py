from io import BytesIO

import chess
import cairosvg
import wandb
from PIL import Image

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from dinora.models.torchnet.resnet import ResNetLight


POSITIONS = [
    {
        'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        'material': '0',
        'text': 'Starting chess position'
    },
    {
        'fen': "4qn1k/2R3pp/p4p2/3Q4/P4N2/6P1/5PBP/2R3K1 w - - 0 31",
        'material': '+13',
        'text': 'White is clearly winning, mate in 6',
    },
    {
        'fen': '4qn1k/p1R3pp/5p2/3Q4/P4N2/6P1/5PBP/2R3K1 b - - 0 30',
        'material': '+13',
        'text': 'Black is clearly losing, mate in 6',
    },
    {
        'fen': 'r1r1q2k/5bpp/5p2/p3n3/P7/6P1/3Q1P1P/2R3K1 w - - 0 31',
        'material': '-11',
        'text': 'White is clearly losing, mate in 10',
    },
    {
        'fen': 'r1r1q2k/p4bpp/5p2/4n3/P7/6P1/3Q1P1P/2R3K1 b - - 0 30',
        'material': '-11',
        'text': 'Black is clearly winning, mate in 7'
    }
]


class SampleGameGenerator(Callback):
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.table = wandb.Table(columns=['moves'])

    def on_validation_end(self, trainer: pl.Trainer, pl_module: ResNetLight) -> None:
        board = chess.Board()
        while board.result() == '*' and board.ply() != 120:
            policy, _ = pl_module.eval_by_network(board)
            bestmove = max(policy, key=lambda k: policy[k])
            board.push(bestmove)
        moves = ' '.join(map(lambda m: m.uci(), board.move_stack))
        trainer.logger.log_text(key='sample_game', columns=['moves'], data=[[moves]])


class BoardsEvaluator(Callback):
    def __init__(self):
        self.positions = POSITIONS
    
    @staticmethod
    def board_to_image(board):
        svg = chess.svg.board(board)
        png_out = BytesIO()
        cairosvg.svg2png(svg.encode(), write_to=png_out)
        image = Image.open(png_out)
        return image
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: ResNetLight) -> None:
        data = []
        COLUMNS = [
            'board', 'fen', 'material',
            'text', 'win_prob', 'draw_prob',
            'lose_prob', 'bestmove'
        ]

        for position in self.positions:
            board = chess.Board(fen=position['fen'])
            wandb_image = wandb.Image(self.board_to_image(board))
            policy, probs = pl_module.eval_by_network(board)
            bestmove = max(policy, key=lambda k: policy[k])

            data.append([
                wandb_image,
                position['fen'],
                position['material'],
                position['text'],
                probs[0],
                probs[1],
                probs[2],
                bestmove.uci(),
            ])
        
        trainer.logger.log_text(key='val_positions', columns=COLUMNS, data=data)
