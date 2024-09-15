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
from transformers import Adafactor
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
from transformers import LlamaForCausalLM 

# Import constants and utility functions specific to the project.
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
    # Print the configuration settings.
    print('\n', '-' * 40, 'Config', '-' * 40)
    print(config)
    
    # Initialize Bittensor wallet with the provided configuration.
    wallet = bt.wallet(config=config)
    # Initialize Bittensor subtensor with the provided configuration.
    subtensor = bt.subtensor(config=config)
    # Retrieve the metagraph for the specified netuid.
    metagraph = subtensor.metagraph(netuid=config.netuid)
    
    # Ensure the wallet's hotkey is registered on the subnet.
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    
    # Get the UID (unique identifier) of the wallet's hotkey in the metagraph.
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    
    # Print initialized objects for debugging purposes.
    print('\n', '-' * 40, 'Objects', '-' * 40)
    print(f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}')
    
    # Check if the bucket committed on-chain matches the configured bucket.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception:
        # If not committed or mismatch, commit the bucket to the chain.
        subtensor.commit(wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket)
    
    # Initialize a gradient scaler for mixed precision training.
    scaler = torch.amp.GradScaler()
    
    # Initialize the model with the specified configuration.
    model = LlamaForCausalLM(config=MODEL_CONFIG)
    
    # Initialize the optimizer (Adafactor) with appropriate hyperparameters.
    optimizer = Adafactor(
        model.parameters(),
        lr=config.learning_rate,
        relative_step=False
    )
    print('Optimizer:', optimizer)
    print('Model:', 'llama', '\n')
    
    # Move the model to the specified device (e.g., 'cuda' or 'cpu').
    model.to(config.device)
    # Set the model to training mode.
    model.train()
    # Enable gradient checkpointing to reduce memory usage during backpropagation.
    model.gradient_checkpointing_enable()
    # TODO: Consider using DeepSpeed or FairScale for further memory optimizations.
    
    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        run = wandb.init(project='cont', resume='allow', name=f'M{my_uid}', config=config)
    
    # Initialize epoch counter and current model metadata.
    n_epochs = 0  # Number of epochs completed.
    current_meta = None  # Metadata of the current model version.
    
    # Start the main training loop.
    while True:
        try:
            # Increment the epoch counter.
            n_epochs += 1
            print('\n', '=' * 40, f'Epoch: {n_epochs}', '=' * 40, '\n')
    
            # Resynchronize the chain state to get the latest metagraph.
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
    
            # Log the miner's incentive value to wandb if enabled.
            if config.use_wandb:
                wandb.log({f"Incentive({my_uid})": float(metagraph.I[my_uid])})
    
            # Iterate over the number of pages to train per epoch.
            for step in range(config.pages_per_epoch):
                # Generate the current training window based on the subtensor block.
                local_pages: List[Tuple[str, int, str]] = SubsetFineWebEdu2Loader.next_pages(
                    offset=subtensor.block * WINDOW_SPEED + 100,  # Offset into the future to avoid overlap with validators.
                    n_pages=WINDOW_SIZE,
                    seed=my_uid  # Seed with miner's UID for consistency.
                )
    
                # Select a random page from the evaluation window for training.
                local_page = random.choice(local_pages)
    
                # Create the dataset for the selected page.
                dataset = SubsetFineWebEdu2Loader(
                    batch_size = config.actual_batch_size,
                    sequence_length = SEQUENCE_LENGTH,
                    pages_info = [ local_page ],
                    tokenizer = TOKENIZER
                )
    
                # Calculate the number of gradient accumulation steps to achieve desired batch size.
                accumulation_steps = config.desired_batch_size // config.actual_batch_size
    
                # Zero the gradients of the optimizer.
                optimizer.zero_grad()
                # Training loop over batches in the dataset.
                for idx, batch in enumerate(dataset):
                    # Convert the batch to a PyTorch tensor and move to the device.
                    input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                    labels = input_ids.clone()  # Clone input_ids to use as labels.
    
                    # Mask the padding tokens in labels by setting them to -100.
                    # This tells the loss function to ignore these positions.
                    labels = torch.where(labels == TOKENIZER.pad_token_id, -100, labels)
    
                    # Forward pass with mixed precision.
                    with torch.amp.autocast( config.device, dtype = torch.bfloat16 ):
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss  # Get the loss value.
                        loss = loss / accumulation_steps  # Normalize loss for gradient accumulation.
    
                    # Backward pass to compute gradients with scaled loss.
                    scaler.scale(loss).backward()
                    # Perform optimizer step after accumulating gradients.
                    if (idx + 1) % accumulation_steps == 0:
                        # Unscale the gradients and perform optimizer step.
                        scaler.step(optimizer)
                        # Update the scaler for next iteration.
                        scaler.update()
                        # Zero the gradients for the next step.
                        optimizer.zero_grad()
    
                    # Log training progress to console.
                    print(f"Batch: {idx + 1}, Loss: {loss.item():.4f}")
    
                    # Log metrics to wandb if enabled.
                    if config.use_wandb:
                        wandb.log({
                            "epoch": n_epochs,
                            "step": step + 1,
                            "batch": idx + 1,
                            "loss": loss.item()
                        })
                    
                    # TODO: Delete unnecessary tensors to free up GPU memory.
                    del input_ids, labels, outputs
                    torch.cuda.empty_cache()
    
            After training, remove the previous model from S3 if it exists.
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
    
            # Optionally, implement a mechanism to ensure the model is uploaded only when improved.
            # For example, keep track of validation loss and upload only if it decreases.
    
        # Handle keyboard interrupts to allow graceful shutdown.
        except (KeyboardInterrupt, SystemExit):
            # Clean up by deleting the model from S3 if it exists.
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
    parser.add_argument('--desired_batch_size', type=int, default=1, help='Desired total batch size for training')
    parser.add_argument('--actual_batch_size', type=int, default=1, help='Actual batch size per step')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.95, help='Beta2 for the optimizer')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.1, help='Weight decay for the optimizer')
    parser.add_argument('--pages_per_epoch', type=int, default=1, help='Number of pages to train per epoch')
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
