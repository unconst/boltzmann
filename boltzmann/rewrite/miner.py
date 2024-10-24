import torch
from boltzmann.rewrite.logger import general_logger

from boltzmann.rewrite.model import MinerSlice, Model


class Miner:
    def __init__(self, model: Model, id: int):
        self.model = model
        self.id = id
        self.data: tuple[torch.Tensor, torch.Tensor] | None = None

    def receive_model_slice(self, slice):
        # Update part of the model with received slice from validator
        self.model.update_params(slice)

    def get_slice_from_indices(self, slice_indices: torch.Tensor) -> MinerSlice:
        # Flatten parameters
        params = torch.nn.utils.parameters_to_vector(
            self.model.torch_model.parameters()
        )
        slice = params[slice_indices]
        general_logger.debug(f"Slice: {slice}")
        return MinerSlice(miner_id=self.id, data=slice)
