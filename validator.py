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

# Instantiate the AWS S3 client.
env_config = {**dotenv_values(".env"), **os.environ}  # Load environment variables from .env file and OS environment.
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')  # Get AWS access key ID.
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')  # Get AWS secret access key.

# Create a boto3 client for S3 with the provided credentials.
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',  # Specify the AWS region.
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
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
            epoch_length = config.blocks_per_uid * config.uids_per_epoch  # Total blocks in an epoch.

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
                size=config.uids_per_epoch,
                replace=config.uids_per_epoch > metagraph.n,
                p=probabilities
            )
            np.random.shuffle(epoch_uids)  # Shuffle the UIDs to distribute them over the epoch.
            epoch_uids = [int(u) for u in epoch_uids]  # Convert UIDs to integers.

            # Function to determine the UID to evaluate at a given block.
            def uid_for_block(block: int) -> int:
                # Calculate the index within the epoch based on the block number.
                current_epoch_index = (block % epoch_length) // config.blocks_per_uid
                # Return the UID corresponding to the calculated index.
                return int(epoch_uids[min(len(epoch_uids) - 1, int(current_epoch_index))])

            # Print epoch information for debugging and tracking.
            print('\n', '=' * 40, f'Epoch: {n_epochs}', '=' * 40, '\n')
            print('uids_per_epoch:', config.uids_per_epoch)
            print('blocks_per_uid:', config.blocks_per_uid)
            print('epoch_length:', epoch_length)
            print('epoch_block:', epoch_block)
            print('epoch_hash:', epoch_hash)
            print('probabilities:', probabilities)
            print('epoch_uids:', epoch_uids)

            # Iterate over each UID to be evaluated in the current epoch.
            for index in range(config.uids_per_epoch):

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

                # Move the model to the specified device (e.g., GPU).
                model.to(config.device)

                # Load the tokenizer for the model.
                tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                    'gpt2', verbose=False, clean_up_tokenization_spaces=True
                )
                tokenizer.pad_token = tokenizer.eos_token  # Set the padding token.

                # Generate evaluation pages based on the current epoch block and UID.
                eval_pages: List[Tuple[str, int, str]] = SubsetFineWebEdu2Loader.next_pages(
                    offset=epoch_block * config.window_speed,
                    n_pages=config.window_size,
                    seed=current_uid
                )

                # Generate holdout pages seeded by the epoch hash to prevent miners from knowing them in advance.
                holdout_pages: List[Tuple[str, int, str]] = SubsetFineWebEdu2Loader.next_pages(
                    offset=epoch_block * config.window_speed,
                    n_pages=config.window_size,
                    seed=epoch_hash
                )

                # Continue evaluating the current UID until it changes.
                while uid_for_block(subtensor.block) == current_uid:

                    # Select random pages from the eval and holdout sets.
                    current_block = subtensor.block
                    eval_page = random.choice(eval_pages)
                    holdout_page = random.choice(holdout_pages)

                    # Create datasets for the eval and holdout pages.
                    eval_dataset = SubsetFineWebEdu2Loader(
                        batch_size=config.batch_size,
                        sequence_length=2048,
                        pages_info=[eval_page],
                        tokenizer=tokenizer
                    )
                    holdout_dataset = SubsetFineWebEdu2Loader(
                        batch_size=config.batch_size,
                        sequence_length=2048,
                        pages_info=[holdout_page],
                        tokenizer=tokenizer
                    )

                    # Initialize lists to store losses.
                    eval_losses = []
                    holdout_losses = []

                    # Evaluate the model on the evaluation dataset.
                    for batch in eval_dataset:
                        # Convert the batch to tensors and move to the device.
                        input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                        labels = input_ids.clone()  # Clone input_ids for labels.
                        # Mask the padding tokens.
                        labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
                        with torch.no_grad():
                            # Forward pass through the model.
                            outputs = model(input_ids=input_ids, labels=labels)
                        # Append the loss to the list.
                        eval_losses.append(outputs.loss.item())
                        # Clean up to free memory.
                        del input_ids, labels, outputs
                        torch.cuda.empty_cache()

                    # Evaluate the model on the holdout dataset.
                    for batch in holdout_dataset:
                        input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
                        with torch.no_grad():
                            outputs = model(input_ids=input_ids, labels=labels)
                        holdout_losses.append(outputs.loss.item())
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
                        'eval_page': eval_page,
                        'holdout_page': holdout_page,
                        'eval_losses': eval_losses,
                        'holdout_losses': holdout_losses,
                        'epoch_uids': epoch_uids,
                    }
                    # Append the event to the history.
                    history[current_uid].append(event)
                    if config.use_wandb:
                        # Log the event to Weights and Biases.
                        wandb.log(event)

                # Save the history to S3 storage after processing the UID.
                save_history(wallet, history, config.bucket, CLIENT)

                # Clean up by deleting the model and tokenizer to free memory.
                model.to('cpu')
                del model
                del tokenizer
                torch.cuda.empty_cache()

            ###############
            ## End Epoch ##
            ###############

            # After each epoch, compute the moving average of losses to set weights.

            # Load the evaluation history from S3.
            evals = load_history(my_uid, metagraph, subtensor, CLIENT)

            # Initialize tensors for eval and holdout weights.
            eval_weights = torch.zeros(metagraph.uids.shape)
            hold_weights = torch.zeros(metagraph.uids.shape)

            # For each UID in the evaluations, compute the moving average losses.
            for uid in evals.keys():
                last_block = None
                moving_hold_loss = 0
                moving_eval_loss = 0
                # Iterate over each sample/event for the UID.
                for sample in evals[uid]:
                    block = int(sample['block'])
                    # Calculate the mean losses for the sample.
                    next_eval_loss = float(np.mean(sample['eval_losses']))
                    next_hold_loss = float(np.mean(sample['holdout_losses']))
                    # Calculate the smoothing factor alpha.
                    if last_block is not None:
                        alpha = config.base_alpha * (block - last_block)
                    else:
                        alpha = config.base_alpha
                    # Update the moving averages.
                    moving_hold_loss = alpha * next_hold_loss + (1 - alpha) * moving_hold_loss
                    moving_eval_loss = alpha * next_eval_loss + (1 - alpha) * moving_eval_loss
                    last_block = block  # Update last_block for the next iteration.
                # Store the final moving averages in the weights tensors.
                eval_weights[int(uid)] = -moving_eval_loss
                hold_weights[int(uid)] = -moving_hold_loss

            # Normalize the weights to be in the range [0,1].
            non_zero_eval_indices = eval_weights.nonzero()
            non_zero_hold_indices = hold_weights.nonzero()
            if len(non_zero_eval_indices[0]) > 0:
                eval_min = eval_weights[non_zero_eval_indices].min()
                eval_max = eval_weights[non_zero_eval_indices].max()
                if eval_max != eval_min:
                    eval_weights[non_zero_eval_indices] = (
                        (eval_weights[non_zero_eval_indices] - eval_min) / (eval_max - eval_min)
                    )
                else:
                    eval_weights[non_zero_eval_indices] = 0.0  # Avoid division by zero.

            if len(non_zero_hold_indices[0]) > 0:
                hold_min = hold_weights[non_zero_hold_indices].min()
                hold_max = hold_weights[non_zero_hold_indices].max()
                if hold_max != hold_min:
                    hold_weights[non_zero_hold_indices] = (
                        (hold_weights[non_zero_hold_indices] - hold_min) / (hold_max - hold_min)
                    )
                else:
                    hold_weights[non_zero_hold_indices] = 0.0  # Avoid division by zero.

            # Normalize the weights to sum to 1, handling division by zero.
            if eval_weights.sum() > 0:
                eval_weights = eval_weights / eval_weights.sum()
            else:
                eval_weights = torch.zeros_like(eval_weights)
            if hold_weights.sum() > 0:
                hold_weights = hold_weights / hold_weights.sum()
            else:
                hold_weights = torch.zeros_like(hold_weights)

            # Combine the eval and holdout weights equally.
            weights = 0.5 * eval_weights + 0.5 * hold_weights

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
            if 'tokenizer' in locals():
                del tokenizer
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
    parser.add_argument('--base_alpha', type=float, default=0.001, help='Block-based moving alpha for setting weights.')
    parser.add_argument('--uids_per_epoch', type=int, default=3, help='Number of miners to evaluate in each epoch.')
    parser.add_argument('--blocks_per_uid', type=int, default=10, help='Blocks spent evaluating each miner.')
    parser.add_argument('--window_size', type=int, default=50, help='Size of the eval window used to evaluate the miner')
    parser.add_argument('--window_speed', type=int, default=5, help='Speed that eval window moves forward across series.')
    parser.add_argument('--temperature', type=int, default=20, help='How steep the exponentiation is.')
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
