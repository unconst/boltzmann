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
import json
import copy
import math
import time
import boto3
import torch
import wandb
import random
import argparse
import traceback
import numpy as np
import bittensor as bt
from tqdm import tqdm
from collections import deque
from dotenv import dotenv_values
from types import SimpleNamespace
from typing import Dict, Optional, List, Tuple
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

from common import get_latest_metadata, download_model, load_history, save_history
from dataset import SubsetFineWebEdu2Loader
from constants import (
    LOCAL_DOMINANCE,
    BASE_ALPHA,
    BLOCKS_PER_UID,
    UIDS_PER_EPOCH,
    WINDOW_SIZE, 
    WINDOW_SPEED, 
    SEQUENCE_LENGTH, 
    TOKENIZER, 
    MODEL_CONFIG, 
    CLIENT
)

# Main function that runs the validator script.
def main(config):
    print('\n', '=' * 40, 'Config', '=' * 40,)
    print(config)  # Print the configuration for debugging.

    # Initialize Bittensor wallet, subtensor, and metagraph.
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)

    # Check if the wallet is registered on the specified subnet.
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    # Get the UID (unique identifier) for the wallet's hotkey.
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    print('\n', '=' * 40, 'Objects', '=' * 40,)
    print(f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}')

    # Assert the chain commitment to ensure the validator's bucket is committed on the chain.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception:
        # If not committed, commit the bucket to the chain.
        subtensor.commit(wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket)

    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        run = wandb.init(project='cont', resume='allow', name=f'V{my_uid}', config=config)

    # Initialize counters for tracking progress.
    n_uids = 0      # Number of UIDs evaluated.
    n_epochs = 0    # Number of epochs completed.
    n_samples = 0   # Number of samples processed.

    # Load the evaluation history from S3 storage.
    history = load_history(my_uid, metagraph, subtensor, CLIENT)

    # Main loop that runs indefinitely until interrupted.
    while True:
        try:
            # Increment the epoch counter.
            n_epochs += 1

            # Reconnect to the subtensor to ensure we have the latest state.
            subtensor = bt.subtensor(config=config)

            # Calculate the length of an epoch in blocks.
            epoch_length = BLOCKS_PER_UID * UIDS_PER_EPOCH # Total blocks in an epoch.

            # Determine the starting block of the current epoch.
            epoch_block = int((subtensor.block / epoch_length) * epoch_length)

            # Get the hash of the block at the start of the epoch.
            epoch_hash = subtensor.get_block_hash(epoch_block)

            # Retrieve the metagraph at the epoch block to get consistent data.
            metagraph = subtensor.metagraph(netuid=config.netuid, block=epoch_block)

            # Seed the random number generator with the epoch hash for reproducibility.
            np.random.seed(int(epoch_hash[:10], 16))

            # Calculate the probability distribution for sampling UIDs based on incentives.
            probabilities = (metagraph.I + (0.001 / float(metagraph.n)))  # Add a small value to avoid zeros.
            probabilities /= probabilities.sum()  # Normalize to make it a probability distribution.

            # Sample UIDs for the current epoch based on the calculated probabilities.
            epoch_uids = np.random.choice(
                len(probabilities),
                size = UIDS_PER_EPOCH,
                replace = UIDS_PER_EPOCH > metagraph.n,
                p = probabilities
            )
            np.random.shuffle(epoch_uids)  # Shuffle the UIDs to distribute them over the epoch.
            epoch_uids = [int(u) for u in epoch_uids]  # Convert UIDs to integers.

            # Function to determine the UID to evaluate at a given block.
            def uid_for_block(block: int) -> int:
                # Calculate the index within the epoch based on the block number.
                current_epoch_index = (block % epoch_length) // BLOCKS_PER_UID
                # Return the UID corresponding to the calculated index.
                return int(epoch_uids[min(len(epoch_uids) - 1, int(current_epoch_index))])

            # Print epoch information for debugging and tracking.
            print('\n', '=' * 40, f'Epoch: {n_epochs}', '=' * 40, '\n')
            print('uids_per_epoch:', UIDS_PER_EPOCH)
            print('blocks_per_uid:', BLOCKS_PER_UID)
            print('epoch_length:', epoch_length)
            print('epoch_block:', epoch_block)
            print('epoch_hash:', epoch_hash)
            print('probabilities:', probabilities)
            print('epoch_uids:', epoch_uids)

            # Iterate over each UID to be evaluated in the current epoch.
            for index in range( UIDS_PER_EPOCH ):

                # Get the current UID to evaluate based on the current block.
                n_uids += 1
                current_uid = uid_for_block(subtensor.block)
                print('\n\n', '-' * 20, f'current_uid: {current_uid}', '-' * 20, '\n')

                # Retrieve the miner's metadata, which contains information about their model.
                miner_meta = get_latest_metadata(current_uid, metagraph, subtensor, CLIENT=CLIENT)
                if miner_meta is None:
                    # If no valid metadata is found, wait until the next UID.
                    print('No valid metadata for uid.')
                    print('Waiting for next valid uid ...')
                    while uid_for_block(subtensor.block) == current_uid:
                        time.sleep(12)  # Sleep for a short period before checking again.
                        subtensor.sync()  # Update the subtensor to get the latest block number.
                    continue  # Skip to the next UID.

                # Download the miner's model using the retrieved metadata.
                try:
                    model = download_model(metadata=miner_meta, device='cpu', CLIENT=CLIENT)
                    if model is None:
                        raise ValueError('Miner model is Non-existent.')
                except Exception as e:
                    print('e:', e)
                    # If the model cannot be downloaded, wait until the next UID.
                    print('Waiting for next valid uid ...')
                    while uid_for_block(subtensor.block) == current_uid:
                        time.sleep(12)
                        subtensor.sync()
                    continue  # Skip to the next UID.
                
                # Check that the model.config matches item by item the values in MODEL_CONFIG
                # Miners must upload models which match the model config as set in the constants.py file.
                try:
                    for key, value in MODEL_CONFIG.to_dict().items():
                        model_value = getattr(model.config, key, None)
                        if model_value != value:
                            raise ValueError(f"Model config mismatch for {key}: expected {value}, got {model_value}")
                except Exception as e:
                    print('Model config is incorrect.')
                    # If the model cannot be downloaded, wait until the next UID.
                    print('Waiting for next valid uid ...')
                    while uid_for_block(subtensor.block) == current_uid:
                        time.sleep(12)
                    continue  # Skip to the next UID.

                # Move the model to the specified device (e.g., GPU).
                model.to(config.device)

                # Generate evaluation pages based on the current epoch block and UID.
                local_pages: List[Tuple[str, int, str]] = SubsetFineWebEdu2Loader.next_pages(
                    offset = epoch_block * WINDOW_SPEED,
                    n_pages = WINDOW_SIZE,
                    seed = current_uid
                )

                # Generate global pages seeded by the epoch hash to prevent miners from knowing them in advance.
                global_pages: List[Tuple[str, int, str]] = SubsetFineWebEdu2Loader.next_pages(
                    offset = epoch_block * WINDOW_SPEED,
                    n_pages = WINDOW_SIZE,
                    seed = random.randint(0, 1000) # Randomly select holdout pages (this can be unqiue per miner.)
                )

                # Continue evaluating the current UID until it changes.
                while uid_for_block(subtensor.block) == current_uid:

                    # Select random pages from the local and global sets.
                    current_block = subtensor.block
                    local_page = random.choice(local_pages)
                    global_page = random.choice(global_pages)

                    # Create datasets for the eval and global pages.
                    local_dataset = SubsetFineWebEdu2Loader(
                        batch_size = config.batch_size,
                        sequence_length = SEQUENCE_LENGTH,
                        pages_info =[local_page],
                        tokenizer = TOKENIZER
                    )
                    global_dataset = SubsetFineWebEdu2Loader(
                        batch_size = config.batch_size,
                        sequence_length = SEQUENCE_LENGTH,
                        pages_info = [global_page],
                        tokenizer = TOKENIZER
                    )

                    # Initialize lists to store losses.
                    local_losses = []
                    global_losses = []

                    # Evaluate the model on the local dataset.
                    for batch in local_dataset:
                        # Convert the batch to tensors and move to the device.
                        input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                        labels = input_ids.clone()  # Clone input_ids for labels.
                        # Mask the padding tokens.
                        labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
                        with torch.no_grad():
                            # Forward pass through the model.
                            outputs = model(input_ids=input_ids, labels=labels)
                        # Append the loss to the list.
                        local_losses.append(outputs.loss.item())
                        # Clean up to free memory.
                        del input_ids, labels, outputs
                        torch.cuda.empty_cache()

                    # Evaluate the model on the global dataset.
                    for batch in global_dataset:
                        input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
                        with torch.no_grad():
                            outputs = model(input_ids=input_ids, labels=labels)
                        global_losses.append(outputs.loss.item())
                        del input_ids, labels, outputs
                        torch.cuda.empty_cache()

                    # Record the evaluation event.
                    n_samples += 1
                    if current_uid not in history:
                        history[current_uid] = []  # Initialize history for the UID if not present.

                    # Create an event dictionary with all relevant information.
                    event = {
                        'uid': current_uid,
                        'n_epochs': n_epochs,
                        'n_samples': n_samples,
                        'epoch_block': epoch_block,
                        'block': current_block,
                        'local_page': local_page,
                        'global_page': global_page,
                        'local_losses': local_losses,
                        'global_losses': global_losses,
                        'epoch_uids': epoch_uids,
                    }
                    # Append the event to the history.
                    history[current_uid].append(event)
                    if config.use_wandb:
                        # Log the event to Weights and Biases.
                        wandb.log(event)

                # Save the history to S3 storage after processing the UID.
                save_history(wallet, history, config.bucket, CLIENT)

                # Clean up by deleting the model to free memory.
                model.to('cpu')
                del model
                torch.cuda.empty_cache()

            ###############
            ## End Epoch ##
            ###############

            # After each epoch, compute the moving average of losses to set weights.

            # Load the evaluation history from S3.
            evals = load_history(my_uid, metagraph, subtensor, CLIENT)

            # Initialize tensors for local and global weights.
            local_weights = torch.zeros(metagraph.uids.shape)
            global_weights = torch.zeros(metagraph.uids.shape)

            # For each UID in the evaluations, compute the moving average losses.
            for uid in evals.keys():
                last_block = None
                moving_global_loss = 0
                moving_local_loss = 0
                # Iterate over each sample/event for the UID.
                for sample in evals[uid]:
                    block = int(sample['block'])
                    # Calculate the mean losses for the sample.
                    next_local_loss = float(np.mean(sample['local_losses']))
                    next_global_loss = float(np.mean(sample['global_losses']))
                    # Calculate the smoothing factor alpha.
                    if last_block is not None:
                        alpha = BASE_ALPHA * (block - last_block)
                    else:
                        alpha = BASE_ALPHA
                    # Update the moving averages.
                    moving_global_loss = alpha * next_global_loss + (1 - alpha) * moving_global_loss
                    moving_local_loss = alpha * next_local_loss + (1 - alpha) * moving_local_loss
                    last_block = block  # Update last_block for the next iteration.
                # Store the negative of the final moving averages in the weights tensors.
                local_weights[int(uid)] = -moving_local_loss
                global_weights[int(uid)] = -moving_global_loss

            # Normalize the weights to be in the range [0,1].
            # This means lower losses will have higher values in the normalization.
            non_zero_local_indices = local_weights.nonzero()
            non_zero_global_indices = global_weights.nonzero()
            if len(non_zero_local_indices[0]) > 0:
                local_min = local_weights[non_zero_local_indices].min()
                local_max = local_weights[non_zero_local_indices].max()
                if local_max != local_min:
                    local_weights[non_zero_local_indices] = (
                        (local_weights[non_zero_local_indices] - local_min) / (local_max - local_min)
                    )
                else:
                    local_weights[non_zero_local_indices] = 0.0  # Avoid division by zero.

            if len(non_zero_global_indices[0]) > 0:
                global_min = global_weights[non_zero_global_indices].min()
                global_max = global_weights[non_zero_global_indices].max()
                if global_max != global_min:
                    global_weights[non_zero_global_indices] = (
                        (global_weights[non_zero_global_indices] - global_min) / (global_max - global_min)
                    )
                else:
                    global_weights[non_zero_global_indices] = 0.0  # Avoid division by zero.

            # Normalize the weights to sum to 1, handling division by zero.
            if local_weights.sum() > 0:
                local_weights = local_weights / local_weights.sum()
            else:
                local_weights = torch.zeros_like(local_weights)
            if global_weights.sum() > 0:
                global_weights = global_weights / global_weights.sum()
            else:
                global_weights = torch.zeros_like(global_weights)

            # Combine the local and global weights equally.
            # The miners must perform well on the global out and local sets.
            weights = LOCAL_DOMINANCE * local_weights + ( 1 - LOCAL_DOMINANCE ) * global_weights

            # Set the computed weights on the chain using the wallet.
            subtensor.set_weights(
                wallet=wallet,
                netuid=metagraph.netuid,
                uids=metagraph.uids.tolist(),
                weights=weights.tolist(),
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )
            print('Weights:', weights.tolist())

        # Handle keyboard interrupts to allow graceful shutdown.
        except (KeyboardInterrupt, SystemExit):
            break

        # Handle any other exceptions, log the error, clean up, and continue.
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            # Clean up resources if they exist.
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
            time.sleep(5)  # Wait for a short period before retrying.
            continue

# Entry point of the script.
if __name__ == "__main__":
    # Create an argument parser for command-line options.
    parser = argparse.ArgumentParser(description='Validator script')

    # Add command-line arguments with default values and help descriptions.
    parser.add_argument('--name', type=str, default=None, help='Optional name')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network uid.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
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
