import time
import typing
from pathlib import Path
from pprint import pprint

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback

import wandb
from elofish.elofish import MatchConfig, PlayerConfig, Rating, run_elo_evaluation


class ElofishRatingEstimator(Callback):
    def __init__(
        self,
        games_count: int,
        step_freq: int,
        time_limit: float,
        rating: float,
        student_deviation: float,
        student_command_prefix: list[str],
        teacher_command: list[str],
        upload_reports: bool,
    ):
        self.games_count = games_count
        self.step_freq = step_freq
        self.time_limit = time_limit
        self.current_rating = rating
        self.student_deviation = student_deviation
        self.student_command_prefix = student_command_prefix
        self.teacher_command = teacher_command
        self.upload_reports = upload_reports

        self.min_rating = 1400
        self.min_start_deviation = 150
        self.current_id = 0
        self.current_deviation = self.student_deviation

    def bulid_match_config(self, weights_path: Path) -> MatchConfig:
        student_command = self.student_command_prefix + [
            "--weights",
            str(weights_path.absolute()),
        ]
        return MatchConfig(
            max_games=self.games_count,
            min_phi=20,
            min_mu=self.min_rating,
            teacher_player=PlayerConfig(
                player_class="StockfishPlayer",
                start_rating=Rating(deviation=10),
                init={
                    "command": self.teacher_command,
                    "options": {},
                    "time_limit": self.time_limit,
                },
            ),
            student_player=PlayerConfig(
                player_class="UCIPlayer",
                start_rating=Rating(
                    rating=self.current_rating, deviation=self.current_deviation
                ),
                init={
                    "command": student_command,
                    "options": {
                        "cpuct": "3.0",
                        "batch_size": "16",
                        "virtual_visits": "1",
                        "max_collisions": "1",
                    },
                    "time_limit": self.time_limit,
                },
            ),
        )

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: typing.Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.step_freq != 0:
            return

        start_time = time.time()
        print(f"Elofish {self.current_id} started")
        weights_path = Path("reports/models/")
        weights_path.mkdir(parents=True, exist_ok=True)
        weights_path = weights_path / "elofish.ckpt"
        pl_module.eval()
        torch.save(pl_module, weights_path)
        pl_module.train()

        if self.current_rating <= self.min_rating:
            self.current_rating = self.min_rating + 1
            self.current_deviation = self.student_deviation

        match_config = self.bulid_match_config(weights_path)
        pprint(match_config)
        evalresult = run_elo_evaluation(match_config)
        self.current_rating, self.current_deviation = (
            float(evalresult.new_rating),
            float(evalresult.new_deviation),
        )

        metrics = {
            "elofish/rating": self.current_rating,
            "elofish/deviation": self.current_deviation,
        }
        print(f"Elofish results: {metrics}")

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics)

        if self.upload_reports and wandb.run is not None:
            elofish_report = wandb.Artifact(
                name=f"elofish{self.current_id}", type="elofish-report"
            )
            elofish_report.add_dir(str(evalresult.report_dir.absolute()))
            wandb.log_artifact(elofish_report)

        self.current_id += 1
        print(
            f"Callback {self.__class__.__name__} took {time.time() - start_time:.3f} seconds"
        )
