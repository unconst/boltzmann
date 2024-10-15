from abc import ABC, abstractmethod
from copy import deepcopy

import torch

from boltzmann.rewrite.validator import Validator
from boltzmann.rewrite.logger import general_logger


class LossCalculator(ABC):
    @abstractmethod
    def calculate(self, validator, miner_slice, data):
        pass


class ExactLoss(LossCalculator):
    def calculate(
        self, validator: Validator, miner_slice: torch.Tensor, data: torch.Tensor
    ):
        # Exact forward pass loss calculation
        loss_before = validator.model.compute_loss(data)

        # Apply miner slice
        validator_params = torch.nn.utils.parameters_to_vector(
            validator.model.torch_model.parameters()
        )
        validator_slice = validator_params[validator.slice_indices]
        new_torch_model = validator.model.aggregate_slices(
            slices=[validator_slice, miner_slice], slice_indices=validator.slice_indices
        )
        new_model = deepcopy(validator.model)
        new_model.torch_model = new_torch_model
        loss_after = new_model.compute_loss(data)
        general_logger.debug(f"Loss before: {loss_before}, loss after: {loss_after}")
        return loss_after - loss_before


class FirstOrderTaylor(LossCalculator):
    def calculate(self, validator, miner_slice, data):
        raise NotImplementedError()


class SecondOrderTaylor(LossCalculator):
    def calculate(self, validator, miner_slice, data):
        raise NotImplementedError()


class CosineSimilarity(LossCalculator):
    def calculate(self, validator, miner_slice, data):
        raise NotImplementedError()
