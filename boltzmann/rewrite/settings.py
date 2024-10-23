from pydantic_settings import BaseSettings
from datetime import datetime
from pathlib import Path
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyNNSettings(BaseSettings):
    hidden_size: int = 10000
    output_size: int = 1
    input_size: int = 1


class GeneralSettings(BaseSettings):
    # Distributed learning
    num_miners: int = 5
    num_communication_rounds: int = 3000
    compression_factor: int = 100

    # Data
    batch_size: int = 128
    num_workers_dataloader: int = 4 if device == "cuda" else 0

    # Synthetic data
    data_size: int = 10000
    val_data_fraction: float = 0.2

    # Results
    results_dir: Path = Path(__file__).parents[2] / "results"


tiny_nn_settings = TinyNNSettings()
general_settings = GeneralSettings()
start_ts = datetime.now()
