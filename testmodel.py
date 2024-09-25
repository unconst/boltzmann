import torch
import numpy as np
from hparams import load_hparams
from transformers import LlamaForCausalLM
import hashlib

# Set up argument parser
hparams = load_hparams()
model = LlamaForCausalLM(config=hparams.model_config)
# Get the shape of all the parameters
param_shapes = {name: param.shape for name, param in model.named_parameters()}

# Produce a hash of the shapes
shapes_str = ''.join([str(shape) for shape in param_shapes.values()])
shapes_hash = hashlib.sha256(shapes_str.encode()).hexdigest()

# Print the hash to the terminal
print(f'Parameter shapes hash: {shapes_hash}')

