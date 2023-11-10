from pathlib import Path

import lightning as l
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, in_paths: list[Path], out_paths: list[Path]) -> None:
        super(__class__, self).__init__()

        self.in_paths = in_paths
        self.out_paths = out_paths

    def __len__(self) -> int:
        return len(self.in_paths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        in_path = self.in_paths[idx]
        out_path = self.out_paths[idx]

        in_data = np.load(in_path)
        out_data = np.load(out_path)

        return in_data, out_data


class DataModule(l.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super(__class__, self).__init__()

        self.in_path_glob = cfg.in_path_glob
        self.out_path_glob = cfg.out_path_glob

        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.split = cfg.split

        self.collate_fn = instantiate(cfg.collate_fn)

    def setup(self, _: str | None = None) -> None:
        in_paths = Path.glob(self.in_path_glob)
        out_paths = Path.glob(self.out_path_glob)

        if len(in_paths) != len(out_paths):
            msg = "Number of input and output files must be the same."
            raise ValueError(msg)

        self.split = np.array(self.split) / sum(self.split) * len(in_paths)
        self.split = np.cumsum(self.split).astype(int)

        self.train_dataset = Dataset(in_paths[: self.split[0]], out_paths[: self.split[0]])
        self.val_dataset = Dataset(in_paths[self.split[0] : self.split[1]], out_paths[self.split[0] : self.split[1]])
        self.test_dataset = Dataset(in_paths[self.split[1] :], out_paths[self.split[1] :])

    @property
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
