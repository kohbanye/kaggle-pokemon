from dataclasses import dataclass


@dataclass
class DataConfig:
    n_samples: int = 1000
    n_features: int = 20
    n_informative: int = 10
    n_classes: int = 3
    test_size: float = 0.2
    batch_size: int = 32
    num_workers: int = 2
    random_state: int = 42


@dataclass
class ModelConfig:
    input_dim: int = 20
    hidden_dim: int = 64
    num_classes: int = 3
    learning_rate: float = 1e-3


@dataclass
class TrainerConfig:
    max_epochs: int = 5
    accelerator: str = "cpu"
    devices: str = "auto"
    log_every_n_steps: int = 10
