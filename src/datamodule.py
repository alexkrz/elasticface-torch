import os

import datasets
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class HFDataset(Dataset):
    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def __init__(
        self,
        parquet_fp: str,
        transform: transforms.Compose = default_transform,
    ):
        # Huggingface's Dataset.from_parquet() automatically splits the parquet file into shards such that they can be better consumed by the system
        self.dataset = datasets.Dataset.from_parquet(parquet_fp, split="train")
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        img = entry["img"]  # PIL Image
        label = entry["label"]  # Python int
        label = torch.tensor(label, dtype=torch.long)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class HFDatamodule(L.LightningDataModule):
    def __init__(
        self,
        data_name: str,
        parquet_fp: str,
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        print(f"Generating HFDataset for stage {stage}")
        self.dataset = HFDataset(
            parquet_fp=self.hparams.parquet_fp,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            # generator=torch.Generator().manual_seed(42),
        )


if __name__ == "__main__":
    data_p = os.environ["DATASET_DIR"] + "/TrainDatasets/parquet-files/casia_webface.parquet"
    dataset = HFDataset(data_p)
    print(dataset[0][0].shape)
