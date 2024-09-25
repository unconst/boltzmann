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


# Check for new optimizer.
            if (
                'optimizer' not in locals() or 
                optimizer == None or 
                prev_learning_rate != hparams.learning_rate or 
                prev_optimizer_beta1 != hparams.optimizer_beta1 or 
                prev_optimizer_beta2 != hparams.optimizer_beta2 or 
                prev_optimizer_weight_decay != hparams.optimizer_weight_decay or 
                prev_cosine_epoch_length != hparams.cosine_epoch_length or 
                prev_eta_min != hparams.eta_min
            ):
                print(f'\nResetting optimizer ...') 
                start_time = time.time()
                prev_learning_rate = hparams.learning_rate,
                prev_optimizer_beta1 = hparams.optimizer_beta1
                prev_optimizer_beta2 = hparams.optimizer_beta2
                prev_optimizer_weight_decay = hparams.optimizer_weight_decay
                prev_cosine_epoch_length = hparams.cosine_epoch_length
                prev_eta_min = hparams.eta_min
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr = hparams.learning_rate,  # Peak learning rate
                    betas = ( hparams.optimizer_beta1, hparams.optimizer_beta2 ), # B1 and B2
                    weight_decay = hparams.optimizer_weight_decay,  # Weight decay
                    foreach = True,  # more memory usage, but faster
                )
                scheduler = CosineAnnealingLR( optimizer, T_max = hparams.cosine_epoch_length, eta_min=hparams.eta_min, last_epoch=-1 )
                print(f'\tResetting optimizer completed in {time.time() - start_time} seconds') 
