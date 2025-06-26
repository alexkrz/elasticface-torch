import logging
import os
import time
from dataclasses import dataclass

from jsonargparse import CLI
from torch.utils.data import DataLoader

from backbones.iresnet import iresnet50, iresnet100
from utils.dataset import HFDataset
from utils.utils_logging import init_logging


@dataclass
class Config:
    data_p: str = os.environ["DATASET_DIR"] + "/TrainDatasets/parquet-files/casia_webface.parquet"
    output_p: str = "output/arcface_r50"
    batch_size: int = 128
    network: str = "iresnet50"
    embedding_size: int = 512
    use_se: bool = False


def main(args: Config):
    rank = 0

    if not os.path.exists(args.output_p) and rank == 0:
        os.makedirs(args.output_p)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, args.output_p)

    train_dataset = HFDataset(args.data_p)
    print(train_dataset[0][0].shape)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # load model
    if args.network == "iresnet100":
        backbone = iresnet100(num_features=args.embedding_size, use_se=args.use_se)
    elif args.network == "iresnet50":
        backbone = iresnet50(dropout=0.4, num_features=args.embedding_size, use_se=args.use_se)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()


if __name__ == "__main__":
    args = CLI(Config, as_positional=False)
    main(args)
