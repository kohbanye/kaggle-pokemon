import lightning as L
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import DataConfig


class ClassificationDataModule(L.LightningDataModule):
    def __init__(self, config: DataConfig) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        x, y = make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_informative=self.config.n_informative,
            n_classes=self.config.n_classes,
            random_state=self.config.random_state,
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        self.train_dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        self.val_dataset = TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
