import time
import typing
from pathlib import Path
from pprint import pprint
from typing import Any

import torch
import wandb

from elofish.elofish import (
    EvaluationResult,
    MatchConfig,
    PlayerConfig,
    Rating,
    run_elo_evaluation,
)
from train.callback import Callback


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

    def prepare_weights(self, pl_module: Any) -> Path:
        weights_dir = Path("reports/models/")
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "elofish.ckpt"
        pl_module.eval()
        torch.save(pl_module, weights_path)
        pl_module.train()
        return weights_path

    def prepare_ratings(self) -> None:
        if self.current_deviation < self.min_start_deviation:
            self.current_deviation = self.min_start_deviation

        start_from_scratch = self.current_rating <= self.min_rating
        if start_from_scratch:
            self.current_rating = self.min_rating + 1
            self.current_deviation = self.student_deviation

    def upload_metrics(self, trainer: Any, evalres: EvaluationResult) -> None:
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
            elofish_report.add_dir(str(evalres.report_dir.absolute()))
            wandb.log_artifact(elofish_report)

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
        trainer: Any,
        pl_module: Any,
        batch: typing.Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.step_freq != 0:
            return

        start_time = time.time()
        print(f"Elofish {self.current_id} started")

        weights_path = self.prepare_weights(pl_module)
        self.prepare_ratings()
        match_config = self.bulid_match_config(weights_path)
        pprint(match_config)
        evalres = run_elo_evaluation(match_config)
        self.current_rating, self.current_deviation = (
            float(evalres.new_rating),
            float(evalres.new_deviation),
        )
        self.upload_metrics(trainer, evalres)

        self.current_id += 1
        print(
            f"Callback {self.__class__.__name__} took {time.time() - start_time:.3f} seconds"
        )
