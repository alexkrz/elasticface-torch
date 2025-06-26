import logging
import os
import time

import torch

from utils.utils_logging import AverageMeter


class CallBackLogging(object):
    def __init__(
        self,
        frequent,
        rank,
        total_step,
        batch_size,
        world_size,
        writer=None,
        resume=0,
        rem_total_steps=None,
    ):
        self.frequent: int = frequent
        self.rank: int = rank
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.writer = writer
        self.resume = resume
        self.rem_total_steps = rem_total_steps

        self.init = False
        self.tic = 0

    def __call__(self, global_step, loss: AverageMeter, epoch: int):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float("inf")

                time_now = (time.time() - self.time_start) / 3600
                # TODO: resume time_total is not working
                if self.resume:
                    time_total = time_now / ((global_step + 1) / self.rem_total_steps)
                else:
                    time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    # self.writer.add_scalar("time_for_end", time_for_end, global_step)
                    self.writer.add_scalar("loss", loss.avg, global_step)
                msg = (
                    "Speed %.2f samples/sec   Loss %.4f Epoch: %d   Global Step: %d   Required: %1.f hours"
                    % (speed_total, loss.avg, epoch, global_step, time_for_end)
                )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


class CallBackModelCheckpoint(object):
    def __init__(self, rank, output="./"):
        self.rank: int = rank
        self.output: str = output

    def __call__(self, global_step, backbone: torch.nn.Module, header: torch.nn.Module = None):
        if global_step > 100 and self.rank == 0:
            torch.save(
                backbone.module.state_dict(),
                os.path.join(self.output, str(global_step) + "backbone.pth"),
            )
        if global_step > 100 and header is not None:
            torch.save(
                header.module.state_dict(),
                os.path.join(self.output, str(global_step) + "header.pth"),
            )
