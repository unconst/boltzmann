# The MIT License (MIT)
# © 2024 Chakana.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import io
import copy
import math
import time
import boto3
import torch
import wandb
import typer
import random
import argparse
import tempfile
import bittensor as bt
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

# Import tooling.
from constants import WINDOW_SIZE, WINDOW_SPEED, SEQUENCE_LENGTH, TOKENIZER, MODEL_CONFIG, CLIENT
from common import upload_model, get_latest_metadata, download_model, hash_model
from dataset import SubsetFineWebEdu2Loader

def main(config):
    """
    Main function for the miner script.

    This function initializes the model, sets up training parameters, and
    enters the main training loop where the model is trained and periodically
    uploaded to the S3 bucket for validation.

    Args:
        config: The configuration object containing training parameters and settings.
    """
    print('\n', '-' * 40, 'Config', '-' * 40)
    print(config)  # Display the configuration settings.

    # Initialize Bittensor objects.
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)

    # Ensure the wallet's hotkey is registered on the subnet.
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')

    # Get the UID (unique identifier) for the wallet's hotkey.
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    print('\n', '-' * 40, 'Objects', '-' * 40)
    print(f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}')

    # Assert the chain commitment to ensure the miner's bucket is committed on the chain.
    try:
        # Check if the bucket committed on-chain matches the configured bucket.
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception:
        # If not committed, commit the bucket to the chain.
        subtensor.commit(wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket)

    # Initialize the model.
    # For better performance on the validation system, we can adjust the model configuration.
    model = LlamaForCausalLM( config = MODEL_CONFIG )

    # Initialize the optimizer with appropriate hyperparameters.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,  # Learning rate.
        betas=(config.optimizer_beta1, config.optimizer_beta2),  # Beta1 and Beta2 for Adam optimizer.
        weight_decay=config.optimizer_weight_decay  # Weight decay for regularization.
    )
    print('Optimizer:', optimizer)
    print('Model:', 'llama', '\n')

    model.to(config.device)  # Move the model to the specified device (CPU or GPU).
    model.train()  # Set the model to training mode.

    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        run = wandb.init(project='cont', resume='allow', name=f'M{my_uid}', config=config)

    # Main training loop variables.
    n_epochs = 0  # Number of epochs completed.
    current_meta = None  # Metadata of the current model version.

    # Main training loop.
    while True:
        try:
            # Increment the epoch counter.
            n_epochs += 1
            print('\n', '=' * 40, f'Epoch: {n_epochs}', '=' * 40, '\n')

            # Resynchronize the chain state to get the latest metagraph.
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)

            # Log the miner's incentive value if wandb is enabled.
            if config.use_wandb:
                wandb.log({f"Incentive({my_uid})": float(metagraph.I[my_uid])})

            # Iterate over the number of pages to train per epoch.
            for step in range(config.pages_per_epoch):
                # Generate the current training window based on the subtensor block.
                eval_pages: List[Tuple[str, int, str]] = SubsetFineWebEdu2Loader.next_pages(
                    offset = subtensor.block * WINDOW_SPEED + 100,  # Offset into the future to avoid overlap with validators.
                    n_pages = WINDOW_SIZE,
                    seed = my_uid  # Seed with miner's UID for consistency.
                )

                # Select a random page from the evaluation window for training.
                selected_page = random.choice(eval_pages)

                # Create the dataset for the selected page.
                dataset = SubsetFineWebEdu2Loader(
                    batch_size = config.batch_size,
                    sequence_length = SEQUENCE_LENGTH,
                    pages_info = [ selected_page ],
                    tokenizer = TOKENIZER
                )

                # Training loop over batches in the dataset.
                for idx, batch in enumerate(dataset):
                    # Convert the batch to a PyTorch tensor and move to the device.
                    input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                    labels = input_ids.clone()  # Clone input_ids for labels.

                    # Mask the padding tokens in labels to ignore them in loss computation.
                    labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)

                    # Forward pass through the model.
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss  # Get the loss value.

                    # Backward pass to compute gradients.
                    loss.backward()

                    # Log training progress.
                    print(f"Epoch: {n_epochs}, Step: {step + 1}/{config.pages_per_epoch}, Batch: {idx + 1}, Loss: {loss.item():.4f}")

                    # Log metrics to wandb if enabled.
                    if config.use_wandb:
                        wandb.log({
                            "epoch": n_epochs,
                            "step": step + 1,
                            "batch": idx + 1,
                            "loss": loss.item()
                        })

                    # Optimizer step to update model parameters.
                    optimizer.step()
                    optimizer.zero_grad()  # Reset gradients.

            # After training, remove the previous model from S3 if it exists.
            if current_meta is not None:
                CLIENT.delete_object(Bucket=config.bucket, Key=current_meta.filename)
                CLIENT.delete_object(Bucket=config.bucket, Key=current_meta.metadata_filename)

            # Upload the current model to S3 for validation.
            current_meta = upload_model(
                wallet=wallet,
                model=model,
                block=int(time.time()),  # Use current timestamp as block number.
                extras={},  # Additional metadata can be added here.
                bucket=config.bucket,
                CLIENT=CLIENT,
            )

            # Optionally, we can implement a mechanism to ensure the model is uploaded only when improved.
            # For example, we can keep track of the validation loss and only upload if it decreases.

        # Handle keyboard interrupts to allow graceful shutdown.
        except (KeyboardInterrupt, SystemExit):
            # Clean up by deleting the model from S3 if it exists.
            if current_meta is not None:
                CLIENT.delete_object(Bucket=config.bucket, Key=current_meta.filename)
                CLIENT.delete_object(Bucket=config.bucket, Key=current_meta.metadata_filename)
            print("Training interrupted. Exiting gracefully.")
            break

        # Handle any other exceptions, log the error, and continue after a short delay.
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            continue

if __name__ == "__main__":
    # Create an argument parser for command-line options.
    parser = argparse.ArgumentParser(description='Miner script')

    # Add command-line arguments with default values and help descriptions.
    parser.add_argument('--name', type=str, default=None, help='Optional miner name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network UID.')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.95, help='Beta2 for the optimizer')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.1, help='Weight decay for the optimizer')
    parser.add_argument('--pages_per_epoch', type=int, default=5, help='Number of pages to train per epoch')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')

    # Add arguments from Bittensor modules for wallet and subtensor configurations.
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)

    # Parse the arguments to create a configuration object.
    config = bt.config(parser)

    # Set the chain endpoint for the subtensor (fixed value).
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'

    # Call the main function with the parsed configuration.
    main(config)
