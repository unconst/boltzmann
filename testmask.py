import torch
import numpy as np
from hparams import load_hparams
from transformers import LlamaForCausalLM
import hashlib
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate mask indices and their hash.')
parser.add_argument('--device', type=str, default='cpu', help='Device to use for computation (default: cpu)')
args = parser.parse_args()
hparams = load_hparams()
model = LlamaForCausalLM(config=hparams.model_config)
for mask_wid in range(1, 2):
    device = args.device
    mask_indices = {}

    # Using NumPy
    np.random.seed(mask_wid)
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
    print(f'NumPy Mask WID {mask_wid} Indices hash: {indices_hash}')

    # Using PyTorch
    mask_indices = {}
    torch.manual_seed(mask_wid)
    if device == 'cuda':
        torch.cuda.manual_seed(mask_wid)  # For CUDA operations if running on GPU
        torch.backends.cudnn.deterministic = True  # Enforce deterministic algorithms in cuDNN
        torch.backends.cudnn.benchmark = False     # Disable cuDNN's auto-tuner that selects the best algorithms

    for name, param in model.named_parameters():
        param = param.to(device)
        next_mask = (torch.rand(param.shape, device=device) < (1 / hparams.compression)).float()
        indices = next_mask.flatten().nonzero(as_tuple=False).flatten()
        mask_indices[name] = indices

    if device == 'cuda':
        torch.backends.cudnn.deterministic = False  # Revert deterministic algorithms in cuDNN
        torch.backends.cudnn.benchmark = True     # Re-enable cuDNN's auto-tuner

    # Produce a hash of the indices
    indices_str = ''.join([str(idx.item()) for indices in mask_indices.values() for idx in indices])
    indices_hash = hashlib.sha256(indices_str.encode()).hexdigest()
    print(f'PyTorch Mask WID {mask_wid} Indices hash: {indices_hash}')