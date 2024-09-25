import torch
import numpy as np
from hparams import load_hparams
from transformers import LlamaForCausalLM
import hashlib

MASK_WID = 1
hparams = load_hparams()
model = LlamaForCausalLM(config=hparams.model_config)

device = 'cpu'

mask_indices = {}
np.random.seed(MASK_WID)
for name, param in model.named_parameters():
    param = param.to(device)
    param_shape = param.shape
    random_values = np.random.rand(*param_shape)  # Generate NumPy random values in [0, 1)
    next_mask = (random_values < (1 / hparams.compression)).astype(np.float32)  # Apply compression ratio
    next_mask_tensor = torch.from_numpy(next_mask).to(device)
    indices = next_mask_tensor.flatten().nonzero(as_tuple=False).flatten()
    mask_indices[name] = indices

# Produce a hash of the indices
indices_str = ''.join([str(idx.item()) for indices in mask_indices.values() for idx in indices])
indices_hash = hashlib.sha256(indices_str.encode()).hexdigest()
print(f'Nump Indices hash: {indices_hash}')