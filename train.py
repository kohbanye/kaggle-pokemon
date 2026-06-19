import logging

import lightning as L

from src.config import DataConfig, ModelConfig, TrainerConfig
from src.data import ClassificationDataModule
from src.model import SimpleClassifier

RANDOM_SEED: int = 42

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    L.seed_everything(RANDOM_SEED)

    data_config = DataConfig(random_state=RANDOM_SEED)
    model_config = ModelConfig()
    trainer_config = TrainerConfig()

    logger.info("Preparing data...")
    data_module = ClassificationDataModule(config=data_config)

    model = SimpleClassifier(
        input_dim=model_config.input_dim,
        hidden_dim=model_config.hidden_dim,
        num_classes=model_config.num_classes,
        learning_rate=model_config.learning_rate,
    )

    trainer = L.Trainer(
        max_epochs=trainer_config.max_epochs,
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        log_every_n_steps=trainer_config.log_every_n_steps,
    )

    logger.info("Starting training...")
    trainer.fit(model, data_module)
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
