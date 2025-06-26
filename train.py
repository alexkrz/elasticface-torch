import os
from dataclasses import dataclass

from jsonargparse import CLI
from torch.utils.data import DataLoader

from utils.dataset import HFDataset


@dataclass
class Config:
    data_p: str = os.environ["DATASET_DIR"] + "/TrainDatasets/parquet-files/casia_webface.parquet"


def main(args: Config):
    train_dataset = HFDataset(args.data_p)
    print(train_dataset[0][0].shape)


if __name__ == "__main__":
    args = CLI(Config, as_positional=False)
    main(args)
