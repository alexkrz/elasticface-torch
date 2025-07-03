import os
import random
from pathlib import Path

import jsonargparse
import lightning as L
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from src.datamodule import HFDatamodule
from src.pl_module import FembModule


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def main(parser: jsonargparse.ArgumentParser):
    cfg = parser.parse_args()
    setup_seed(cfg.seed)

    datamodule = HFDatamodule(**cfg.datamodule)

    pl_module = FembModule(**cfg.pl_module)

    # Modify checkpoint callback
    cfg_callbacks = cfg.trainer.pop("callbacks")
    cp_callback = ModelCheckpoint(
        save_weights_only=True,
        save_last=None,
        monitor="loss",
        mode="min",
        every_n_train_steps=50,  # By default lightning only monitors the metric per epoch
        filename="{epoch}_{step}_{loss:.2f}",
    )

    # Configure loggers
    cfg_logger = cfg.trainer.pop("logger")
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name=pl_module.hparams.header)
    csv_logger = CSVLogger(
        save_dir="lightning_logs", name=pl_module.hparams.header, version=tb_logger.version
    )

    trainer = Trainer(**cfg.trainer, callbacks=[cp_callback], logger=[tb_logger, csv_logger])

    print(f"Writing logs to {trainer.log_dir}")
    Path(trainer.log_dir).mkdir(parents=True)
    parser.save(cfg, Path(trainer.log_dir) / "config.yaml")

    # Prepare datamodule before calling tainer.fit() to check correct behavior
    datamodule.setup("fit")

    trainer.fit(model=pl_module, datamodule=datamodule)


if __name__ == "__main__":
    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(parser_mode="omegaconf")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_class_arguments(
        HFDatamodule,
        "datamodule",
        default={
            "data_name": "webface",
            "parquet_fp": os.environ["DATASET_DIR"]
            + "/TrainDatasets/parquet-files/casia_webface.parquet",
            "num_workers": 8,
        },
    )
    parser.add_class_arguments(
        FembModule,
        "pl_module",
        default={
            "header": "arcface",
            "s": 64.0,
            "m": 0.5,
        },
    )
    parser.link_arguments("datamodule.data_name", "pl_module.data_name", apply_on="parse")
    parser.add_class_arguments(
        Trainer,
        "trainer",
        default={
            "precision": "16-mixed",
            "log_every_n_steps": 50,
            "enable_checkpointing": True,
            "max_steps": 210,
            # "max_epochs": 2,
        },
    )

    main(parser)
