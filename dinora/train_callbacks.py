from io import BytesIO

import chess
import cairosvg
import wandb
from PIL import Image

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from dinora.models.torchnet.resnet import ResNetLight
from dinora.handmade_val_dataset.dataset import POSITIONS


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
    def __init__(self, render_image = False):
        self.positions = POSITIONS
        self.render_image = render_image
    
    @staticmethod
    def board_to_image(board):
        svg = chess.svg.board(board)
        png_out = BytesIO()
        cairosvg.svg2png(svg.encode(), write_to=png_out)
        image = Image.open(png_out)
        return image
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: ResNetLight) -> None:
        data = []
        COLUMNS = ['image'] * self.render_image + [
            'fen', 'text', 'type',
            'stockfish_cp', 'stockfish_wdl', 'stockfish_top3_lines',
            'model_v', 'model_wdl', 'bestmove'
        ]

        for position in self.positions:
            board = chess.Board(fen=position['fen'])
            policy, probs = pl_module.eval_by_network(board)  # TODO: eval in batch
            model_wdl = f"{probs[0]:.2f} | {probs[1]:.2f} | {probs[2]:.2f}"
            bestmove = max(policy, key=lambda k: policy[k])

            entry = [
                position['fen'], position['text'], position['type'],
                position['stockfish_cp'], position['stockfish_wdl'], position['stockfish_top3_lines'],
                probs[0] - probs[2], model_wdl, bestmove.uci(),
            ]
            if self.render_image:
                entry = [wandb.Image(self.board_to_image(board))] + entry
            
            data.append(entry)
        
        trainer.logger.log_text(key='val_positions', columns=COLUMNS, data=data)
