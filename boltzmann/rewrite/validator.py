from collections import defaultdict
from typing import Any

import numpy as np
import torch

from boltzmann.rewrite.logger import general_logger
from boltzmann.rewrite.model import MinerSlice, Model
from boltzmann.rewrite.settings import general_settings


class Validator:
    def __init__(
        self,
        model: Model,
        compression_factor: int,
        loss_calculator: Any,
    ):
        self.model = model
        self.compression_factor = compression_factor
        self.loss_calculator = loss_calculator
        self.scores = defaultdict(lambda: torch.tensor([]))
        self.slices = {}
        self.slice_indices = torch.Tensor([])
        self.reset_slices_and_indices()

    def add_miner_slice(self, miner_slice: MinerSlice) -> None:
        self.slices.update({miner_slice.miner_id: miner_slice.data})

    def reset_slices_and_indices(self):
        # Reset slices
        self.slices = {}

        # Calculate slice size
        num_params_total = self.model.num_params()
        slice_size = num_params_total // general_settings.compression_factor

        # Define indices
        self.slice_indices = torch.tensor(
            np.random.choice(num_params_total, slice_size, replace=False)
        )
        general_logger.debug(f"Total number of parameters: {num_params_total}")
        general_logger.debug(
            f"Selected indices after compressing {general_settings.compression_factor}x: {self.slice_indices}"
        )

    def _zero_model(self, model: Model) -> Model:
        with torch.no_grad():  # Disables gradient tracking to avoid errors
            for param in model.torch_model.parameters():
                param.zero_()  # Set the parameter tensor values to zero
        return model

    def calculate_miner_score(self, miner_slice: torch.Tensor, miner_id: int, data):
        new_score = self.loss_calculator.calculate(self, miner_slice, data)
        new_score = torch.Tensor([new_score])
        self.scores[miner_id] = torch.cat([self.scores[miner_id], new_score])
