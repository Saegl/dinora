import pathlib
import time
from io import BytesIO
from typing import Any

import cairosvg
import chess
import chess.svg
import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from PIL import Image

from cploss.evaluate import calc_policy_cploss, calc_value_cploss, load_cploss
from dinora import PROJECT_ROOT
from train.handmade_val_dataset.dataset import POSITIONS


class SampleGameGenerator(Callback):
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.table = wandb.Table(columns=["moves"])  # type: ignore

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        start_time = time.time()
        board = chess.Board()
        while board.result() == "*" and board.ply() != 120:
            policy, _ = pl_module.evaluate(board)
            bestmove = max(policy, key=lambda k: policy[k])
            board.push(bestmove)
        moves = " ".join(map(lambda m: m.uci(), board.move_stack))
        trainer.logger.log_text(key="sample_game", columns=["moves"], data=[[moves]])  # type: ignore
        print(
            f"Callback {self.__class__.__name__} took {time.time() - start_time:.3f} seconds"
        )


class BoardsEvaluator(Callback):
    def __init__(self, render_image: bool = False) -> None:
        self.positions = POSITIONS
        self.render_image = render_image

    @staticmethod
    def board_to_image(board: chess.Board) -> Any:
        svg = chess.svg.board(board)
        png_out = BytesIO()
        cairosvg.svg2png(svg.encode(), write_to=png_out)
        image = Image.open(png_out)
        return image

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        start_time = time.time()
        data = []
        COLUMNS = ["image"] * self.render_image + [
            "fen",
            "text",
            "type",
            "stockfish_cp",
            "stockfish_wdl",
            "stockfish_top3_lines",
            "model_v",
            "model_bestmove",
        ]

        for position in self.positions:
            board = chess.Board(fen=position["fen"])
            policy, value = pl_module.evaluate(board)  # TODO: eval in batch
            bestmove = max(policy, key=lambda k: policy[k])

            entry = [
                position["fen"],
                position["text"],
                position["type"],
                position["stockfish_cp"],
                position["stockfish_wdl"],
                position["stockfish_top3_lines"],
                value,
                bestmove.uci(),
            ]
            if self.render_image:
                entry = [wandb.Image(self.board_to_image(board))] + entry

            data.append(entry)

        trainer.logger.log_text(key="val_positions", columns=COLUMNS, data=data)  # type: ignore
        print(
            f"Callback {self.__class__.__name__} took {time.time() - start_time:.3f} seconds"
        )


class ValidationCheckpointer(Callback):
    def __init__(self) -> None:
        self.saves_counter = 0

    def save_model(self, pl_module: pl.LightningModule, label: str) -> None:
        self.saves_counter += 1
        is_module_training = pl_module.training

        filepath = pathlib.Path(f"{label}.ckpt").absolute()

        if is_module_training:
            pl_module.eval()

        torch.save(pl_module, filepath)

        if is_module_training:
            pl_module.train()

        import wandb

        final_state = wandb.Artifact(name=label, type="valid-state")
        final_state.add_file(filepath)  # type: ignore
        wandb.log_artifact(final_state)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        start_time = time.time()
        self.save_model(pl_module, f"valid-state-{self.saves_counter}")
        print(
            f"Callback {self.__class__.__name__} took {time.time() - start_time:.3f} seconds"
        )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.save_model(pl_module, "valid-state-final")


class TrainerCheckpointer(Callback):
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ckpt_filepath = pathlib.Path("trainer.ckpt")
        trainer.save_checkpoint(ckpt_filepath)

        final_state = wandb.Artifact(
            name=f"trainer_checkpoint_{trainer.global_step}", type="trainer_checkpoint"
        )
        final_state.add_file(str(ckpt_filepath.absolute()))
        wandb.log_artifact(final_state)


class CPLoss(Callback):
    def __init__(self, cploss_label: str, max_positions: int, batch_size: int):
        cploss_folder = self.download_from_wandb(cploss_label)
        self.value_boards, self.policy_boards, self.positions = load_cploss(
            cploss_folder, max_positions
        )
        self.batch_size = batch_size

    def download_from_wandb(self, cploss_label: str) -> pathlib.Path:
        folder_name = cploss_label.replace(":", "-").replace("/", "-")
        cploss_folder = PROJECT_ROOT / "data/cploss" / folder_name

        is_wandb_offline = wandb.run and wandb.run.offline
        cached_data_exists = cploss_folder.exists()

        if is_wandb_offline and not cached_data_exists:
            raise Exception("Wandb in offline mode and there is no cached cploss data")
        elif is_wandb_offline and cached_data_exists:
            return cploss_folder
        else:
            assert wandb.run, (
                "Wandb run must be initialized to sync cploss data,"
                "run `wandb offline` if you have cached cploss data"
            )
            cploss_folder.mkdir(parents=True, exist_ok=True)
            dataset_artifact = wandb.run.use_artifact(cploss_label)
            dataset_artifact.download(root=cploss_folder)
            return cploss_folder

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        start_time = time.time()
        if trainer.logger is None:
            print("Cant log cploss metrics")
            return

        policy_cploss = calc_policy_cploss(
            pl_module, self.positions, self.policy_boards, self.batch_size
        )
        value_cploss = calc_value_cploss(
            pl_module, self.positions, self.value_boards, self.batch_size
        )
        metrics = {
            "cploss/policy_mean": float(np.mean(policy_cploss)),
            "cploss/policy_std": float(np.std(policy_cploss)),
            "cploss/policy_max": float(np.max(policy_cploss)),
            "cploss/policy_top0cp": float(np.mean(policy_cploss <= 0.0) * 100),
            "cploss/policy_top50cp": float(np.mean(policy_cploss <= 50.0) * 100),
            "cploss/value_mean": float(np.mean(value_cploss)),
            "cploss/value_std": float(np.std(value_cploss)),
            "cploss/value_max": float(np.max(value_cploss)),
            "cploss/value_top0cp": float(np.mean(value_cploss <= 0.0) * 100),
            "cploss/value_top50cp": float(np.mean(value_cploss <= 50.0) * 100),
        }
        print(f"CPLoss: {metrics}")
        trainer.logger.log_metrics(metrics)
        print(
            f"Callback {self.__class__.__name__} took {time.time() - start_time:.3f} seconds"
        )
