import lightning as l
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


class LitModule(l.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.model = instantiate(cfg.model.instance)
        self.learning_rate = cfg.model.learning_rate

        self.optimizer = instantiate(cfg.model.optimizer, params=self.parameters(), lr=self.learning_rate)
        self.scheduler = instantiate(cfg.model.scheduler, optimizer=self.optimizer)

    def forward(self, **x) -> torch.Tensor:
        return self.model(**x)

    def configure_optimizers(self) -> tuple[list, list]:
        return [self.optimizer], [self.scheduler]
