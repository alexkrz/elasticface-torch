import logging
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from jsonargparse import CLI
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backbones.iresnet import iresnet50, iresnet100
from utils import losses
from utils.dataset import HFDataset
from utils.utils_callbacks import CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging


@dataclass
class Config:
    # TODO: Make config conditional depending on dataset and loss
    seed: int = 42
    dataset: str = "webface"
    data_p: str = os.environ["DATASET_DIR"] + "/TrainDatasets/parquet-files/casia_webface.parquet"
    output_p: str = "output/arcface_r50"
    batch_size: int = 128
    network: str = "iresnet50"
    use_se: bool = False
    embedding_size: int = 512
    loss: str = "ArcFace"
    num_classes: int = 10572
    s: float = 64.0
    m: float = 0.50
    lr: float = 0.1
    weight_decay: float = 5e-4
    warmup_epoch: int = -1
    num_epoch: int = 26
    global_step: int = 0


def add_version_dir(output_p: str):
    # Find existing version_X directories
    if os.path.exists(output_p):
        versions = [
            int(d.split("_")[1])
            for d in os.listdir(output_p)
            if os.path.isdir(os.path.join(output_p, d))
            and d.startswith("version_")
            and d.split("_")[1].isdigit()
        ]
        next_version = max(versions) + 1 if versions else 1
    else:
        next_version = 0
    version_p = os.path.join(output_p, f"version_{next_version}")

    return version_p


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


def main(cfg: Config):
    rank = 0
    world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    version_p = add_version_dir(cfg.output_p)
    setup_seed(cfg.seed, cuda_deterministic=True)

    if not os.path.exists(version_p) and rank == 0:
        os.makedirs(version_p)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, version_p)
    writer = SummaryWriter(version_p)

    trainset = HFDataset(cfg.data_p)

    train_loader = DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
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
    backbone.to(device)
    backbone.train()

    # get header
    if cfg.loss == "ArcFace":
        header = losses.ArcFace(
            in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m
        )
    else:
        print("Header not implemented")
    header.to(device)
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

    criterion = CrossEntropyLoss()

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0:
        logging.info("Total Step is: %d" % total_step)

    callback_logging = CallBackLogging(
        50,
        rank,
        total_step,
        cfg.batch_size,
        world_size,
        writer=writer,
    )
    callback_checkpoint = CallBackModelCheckpoint(rank, version_p)

    loss = AverageMeter()
    global_step = cfg.global_step
    for epoch in range(start_epoch, cfg.num_epoch):
        for _, (img, label) in enumerate(train_loader):
            global_step += 1
            img = img.to(device)
            label = label.to(device)

            features = F.normalize(backbone(img))

            thetas = header(features, label)
            loss_v = criterion(thetas, label)
            loss_v.backward()

            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_header.step()

            opt_backbone.zero_grad()
            opt_header.zero_grad()

            loss.update(loss_v.item(), 1)

            callback_logging(global_step, loss, epoch)

        scheduler_backbone.step()
        scheduler_header.step()

        callback_checkpoint(global_step, backbone, header)

    writer.close()


if __name__ == "__main__":
    args = CLI(Config, as_positional=False)
    main(args)
