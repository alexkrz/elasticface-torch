import logging
import os
import time
from dataclasses import dataclass

import torch
from jsonargparse import CLI
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from backbones.iresnet import iresnet50, iresnet100
from utils import losses
from utils.dataset import HFDataset
from utils.utils_logging import init_logging


@dataclass
class Config:
    dataset: str = "webface"
    data_p: str = os.environ["DATASET_DIR"] + "/TrainDatasets/parquet-files/casia_webface.parquet"
    output_p: str = "output/arcface_r50"
    batch_size: int = 128
    network: str = "iresnet50"
    use_se: bool = False
    embedding_size: int = 512
    num_classes: int = 10572
    s: float = 64.0
    m: float = 0.50
    lr: float = 0.1
    weight_decay: float = 5e-4
    warmup_epoch: int = -1


def main(cfg: Config):
    rank = 0
    world_size = 1

    if not os.path.exists(cfg.output_p) and rank == 0:
        os.makedirs(cfg.output_p)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output_p)

    train_dataset = HFDataset(cfg.data_p)
    print(train_dataset[0][0].shape)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # load model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.use_se)
    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=0.4, num_features=cfg.embedding_size, use_se=cfg.use_se)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()
    backbone.train()

    # get header
    if cfg.loss == "ArcFace":
        header = losses.ArcFace(
            in_features=cfg.embedding_size,
            out_features=cfg.num_classes,
            s=cfg.s,
            m=cfg.m,
        )
    else:
        print("Header not implemented")
    header.train()

    opt_backbone = torch.optim.SGD(
        params=[{"params": backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    opt_header = torch.optim.SGD(
        params=[{"params": header.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )

    if cfg.dataset == "webface":

        def lr_step_func(epoch):
            return (
                ((epoch + 1) / (4 + 1)) ** 2
                if epoch < cfg.warmup_epoch
                else 0.1 ** len([m for m in [22, 30, 40] if m - 1 <= epoch])
            )
    else:
        raise NotImplementedError("Unknown datase")

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=lr_step_func
    )
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=lr_step_func
    )


if __name__ == "__main__":
    args = CLI(Config, as_positional=False)
    main(args)
