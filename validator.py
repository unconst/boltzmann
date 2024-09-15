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
    EPOCHS_PER_SET_WEIGHTS,
    EPOCH_CLIFF,
    BASE_PROBABILITY,
    TEMPERATURE,
    LOCAL_DOMINANCE,
    BASE_ALPHA,
    BLOCKS_PER_EPOCH,
    WINDOW_SIZE, 
    WINDOW_SPEED, 
    SEQUENCE_LENGTH, 
    TOKENIZER, 
    MODEL_CONFIG, 
    CLIENT
)

# Main function that runs the validator script.
def main(config):
    # Print the configuration for debugging.
    print('\n', '=' * 40, 'Config', '=' * 40)
    print(config)

    # Initialize Bittensor wallet, subtensor, and metagraph.
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)

    # Check if the wallet is registered on the specified subnet.
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    # Get the UID (unique identifier) for the wallet's hotkey.
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

    print('\n', '=' * 40, 'Objects', '=' * 40)
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

    # Load the evaluation history from S3 storage.
    history = load_history(my_uid, metagraph, subtensor, CLIENT)
    # Optionally clear history on restart.
    if config.restart:
        history = {}
        save_history(wallet, history, config.bucket, CLIENT)

    # Main loop that runs indefinitely until interrupted.
    skip_uid = False    # Whether the script should skip the current UID.
    last_epoch = None   # Block of last UID change.
    model = None        # Model to evaluate on step.
    while True:
        try:
            # Reconnect to the subtensor to ensure we have the latest state.
            subtensor = bt.subtensor(config=config)

            # Determine current block and compute the epoch block.
            block = subtensor.block
            
            # Compute the epoch block (start of the current epoch).
            epoch = int((block / BLOCKS_PER_EPOCH))
            epoch_block = epoch * BLOCKS_PER_EPOCH

            # If the epoch has changed, select the next UID to evaluate.
            # Skip the current UID until the epoch has changed.
            if epoch_block == last_epoch and skip_uid:
                time.sleep(6)
                continue
            
            # This section checks if the epoch has changed. When the epoch ends, the validator script
            # Moved to the next UID to evaluate. This is computed based on the hash of the block at the epoch
            # All validators will select the same UID to evaluate at this moment giving determinism to the scoring.
            # When a new model is selected we pull their model and begin evaluating on the local and global samples.
            # Note we use the incentive vector as the UID sampling method.
            elif last_epoch is None or epoch_block != last_epoch:
                
                # Increment the epoch counter.
                skip_uid = False
                last_epoch = epoch_block
                
                # Save the history to S3 at the end of the previous epoch.
                save_history(wallet, history, config.bucket, CLIENT)

                # Get the hash of the block at the start of the epoch.
                epoch_hash = subtensor.get_block_hash(epoch_block)

                # Seed the random number generator with the epoch hash for reproducibility.
                np.random.seed(int(epoch_hash[:10], 16))

                # Retrieve the metagraph at the epoch block to get consistent data.
                metagraph = subtensor.metagraph(netuid=config.netuid, block=epoch_block)

                # Determine the probability of selecting each UID based on the incentive at the epoch block.
                probabilities = metagraph.I + (BASE_PROBABILITY / float(metagraph.n))
                probabilities /= probabilities.sum()

                # Sample a single UID to evaluate based on the block probabilities.
                epoch_uid = int(np.argmax(np.random.multinomial(1, probabilities)))

                # Initialize history for the UID if not present.
                if epoch_uid not in history:
                    history[epoch_uid] = []
                    
                # Print epoch information for debugging and tracking.
                print( '\n', '=' * 40, f'Epoch: {epoch}', '=' * 40 )
                print( 'uid:', epoch_uid )
                print( 'block:', block )
                print( 'epoch_block:', epoch_block )
                print( 'next_epoch:', epoch_block + BLOCKS_PER_EPOCH )
                print( 'probabilities:', probabilities )
                print( 'epoch_hash:', epoch_hash )
                print( 'hotkey:', metagraph.hotkeys[epoch_uid] )

                # Retrieve the miner's metadata, which contains information about their model.
                miner_meta = get_latest_metadata(epoch_uid, metagraph, subtensor, CLIENT=CLIENT)
                if miner_meta is None:
                    # If no valid metadata is found, wait until the next UID.
                    print('No valid metadata for uid. Waiting for next valid uid ...')
                    skip_uid = True
                    continue  # Skip to the next UID.
                
                # Clean up previous model if it exists.
                if 'model' in locals() and model is not None:
                    model.to('cpu')
                    del model
                    torch.cuda.empty_cache()

                # Download the miner's model using the retrieved metadata.
                try:
                    model = download_model(metadata=miner_meta, device='cpu', CLIENT=CLIENT)
                    if model is None:
                        raise ValueError('Miner model is Non-existent.')
                except Exception as e:
                    print('Error downloading model:', e, 'Waiting for next valid uid ...')
                    skip_uid = True
                    continue  # Skip to the next UID.

                # Check that the model.config matches item by item the values in MODEL_CONFIG.
                # Miners must upload models which match the model config as set in the constants.py file.
                try:
                    for key, value in MODEL_CONFIG.to_dict().items():
                        model_value = getattr(model.config, key, None)
                        if model_value != value:
                            raise ValueError(f"Model config mismatch for {key}: expected {value}, got {model_value}")
                except Exception as e:
                    print('Model config is incorrect:', e ,'Waiting for next valid uid ...')
                    skip_uid = True
                    continue  # Skip to the next UID.

                # Move the model to the specified device (e.g., GPU).
                model.to(config.device)
                
            # This section generates evaluation pages based on the current epoch block and UID.
            # It calculates the offset as epoch_block * WINDOW_SPEED and selects WINDOW_SIZE pages starting from this offset.
            # The selected pages are used to create a local dataset, which is then evaluated by the model.
            # The losses from the evaluation are collected in the local_losses list. The miner has full knowledge of which pages 
            # Will be selected for local evaluation given the determinism of the function next_pages seeded by the UID and block.
            local_losses = []
            # Select a set of pages for the miner based on their current window.
            local_pages = SubsetFineWebEdu2Loader.next_pages(
                offset = epoch_block * WINDOW_SPEED,
                n_pages = WINDOW_SIZE,
                seed = epoch_uid
            )
            # Randomly select a page from the list.
            local_page = random.choice(local_pages)
            # Create the local dataset.
            local_dataset = SubsetFineWebEdu2Loader(
                batch_size = config.batch_size,
                sequence_length = SEQUENCE_LENGTH,
                pages_info = [local_page],
                tokenizer = TOKENIZER
            )
            # Evaluate the model on the local dataset.
            for batch in local_dataset:
                # Convert the batch to tensors and move to the device.
                input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                labels = input_ids.clone()  # Clone input_ids for labels.
                # Mask the padding tokens.
                labels = torch.where(labels == TOKENIZER.pad_token_id, -100, labels)
                with torch.no_grad():
                    # Forward pass through the model.
                    outputs = model(input_ids=input_ids, labels=labels)
                # Append the loss to the list.
                local_losses.append(outputs.loss.item())
                # Clean up to free memory.
                del input_ids, labels, outputs
                torch.cuda.empty_cache()

            # This section selects a completely random global page to evaluate the miner on.
            # It pulls 1 page from the full 51 million pages available in the dataset.
            # The evaluation is performed on this single page, and the loss is calculated.
            # Note miners will have no knowledge of which global page they will be evaluated on
            # These loss terms are still used to compute weights which means miners must perform 
            # well on average across the entire dataset.
            global_losses = []
            global_dataset = SubsetFineWebEdu2Loader(
                batch_size=config.batch_size,
                sequence_length=SEQUENCE_LENGTH,
                num_pages=1,
                tokenizer=TOKENIZER
            )
            global_page = global_dataset.pages[0]
            # Evaluate the model on the global dataset.
            for batch in global_dataset:
                input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                labels = input_ids.clone()
                labels = torch.where(labels == TOKENIZER.pad_token_id, -100, labels)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, labels=labels)
                global_losses.append(outputs.loss.item())
                del input_ids, labels, outputs
                torch.cuda.empty_cache()

            # Record the evaluation event.
            # Create an event dictionary with all relevant information.
            event = {
                'block': int(subtensor.block),
                'epoch_uid': int(epoch_uid),
                'epoch_block': int(epoch_block),
                'local_page': int(local_page[1]),
                'global_page': int(global_page[1]),
                'local_loss': float(np.mean(local_losses)),
                'global_loss': float(np.mean(global_losses)),
                'local_losses': [float(v) for v in local_losses],
                'global_losses': [float(v) for v in global_losses],
            }
            print ( 'uid', epoch_uid, 'block', block, 'local_page', int(local_page[1]), 'local_loss', float(np.mean(local_losses)), 'global_page', int(global_page[1]), 'global_loss', float(np.mean(global_losses)) )
            # Append the event to the history.
            history[int(epoch_uid)].append(event)
            if config.use_wandb:
                # Log the event to Weights and Biases.
                wandb.log(event)
                
            # Save the history to S3 at the end of the previous epoch.
            print (history)
            save_history(wallet, history, config.bucket, CLIENT)
                
            ###############
            ## End Epoch ##
            ###############

            # Load the evaluation history from S3.
            if epoch % EPOCHS_PER_SET_WEIGHTS == 0:
                # This section calculates the weights for each UID based on their performance.
                # It computes the moving average of local and global losses for each UID.
                # The weights are normalized to be in the range [0,1], with lower losses resulting in higher weights.
                # The weights are then combined using LOCAL_DOMINANCE and adjusted by a temperature factor.
                # Finally, the weights are normalized to sum to 1 and set on the chain.
                evals = load_history(my_uid, metagraph, subtensor, CLIENT)

                # Initialize tensors for local and global weights.
                local_weights = torch.zeros(metagraph.uids.shape)
                global_weights = torch.zeros(metagraph.uids.shape)

                # For each UID in the evaluations, compute the moving average losses.
                # The moving average is calculated using an exponential smoothing formula:
                # moving_loss = alpha * next_loss + (1 - alpha) * moving_loss
                # where alpha is the smoothing factor, which is adjusted based on the block difference.
                # The final moving averages are stored as negative values in the weights tensors.
                for uid in evals.keys():
                    last_block = None
                    moving_global_loss = 0.0
                    moving_local_loss = 0.0
                    # Iterate over each sample/event for the UID.
                    for sample in evals[uid]:
                        try:
                            block = int(sample['block'])
                            # Calculate the mean losses for the sample.
                            next_local_loss = float(np.mean(sample['local_losses']))
                            next_global_loss = float(np.mean(sample['global_losses']))
                            # Calculate the smoothing factor alpha.
                            alpha = BASE_ALPHA * (block - last_block) if last_block is not None else BASE_ALPHA
                            # Update the moving averages.
                            moving_global_loss = alpha * next_global_loss + (1 - alpha) * moving_global_loss
                            moving_local_loss = alpha * next_local_loss + (1 - alpha) * moving_local_loss
                            last_block = block  # Update last_block for the next iteration.
                        except: continue
                    # Store the negative of the final moving averages in the weights tensors.
                    local_weights[int(uid)] = -moving_local_loss
                    global_weights[int(uid)] = -moving_global_loss

                # This section normalizes the local and global weights to be in the range [0,1].
                # The normalization process ensures that lower losses will have higher values after normalization.
                # For each non-zero weight, the normalization is done using the formula:
                # normalized_weight = (weight - min_weight) / (max_weight - min_weight)
                # where min_weight and max_weight are the minimum and maximum weights in the non-zero weights.
                # Find indices of non-zero local and global weights.
                non_zero_local_indices = local_weights.nonzero(as_tuple=True)
                non_zero_global_indices = global_weights.nonzero(as_tuple=True)
                # Normalize local weights if there are non-zero weights.
                if len(non_zero_local_indices[0]) > 0:
                    local_min = local_weights[non_zero_local_indices].min()
                    local_max = local_weights[non_zero_local_indices].max()
                    if local_max != local_min:
                        local_weights[non_zero_local_indices] = (
                            (local_weights[non_zero_local_indices] - local_min) / (local_max - local_min)
                        )
                # Normalize global weights if there are non-zero weights.
                if len(non_zero_global_indices[0]) > 0:
                    global_min = global_weights[non_zero_global_indices].min()
                    global_max = global_weights[non_zero_global_indices].max()
                    if global_max != global_min:
                        global_weights[non_zero_global_indices] = (
                            (global_weights[non_zero_global_indices] - global_min) / (global_max - global_min)
                        )
                        
                # Normalize the weights to sum to 1, handling division by zero.
                # This section normalizes the local and global weights so that their sums equal 1.
                # The normalization is done by dividing each weight by the total sum of weights.
                # Mathematically, for a set of weights w_i, the normalized weight w'_i is calculated as:
                # w'_i = w_i / sum(w_i)
                # This ensures that the sum of all normalized weights is 1.
                total_local_weight = local_weights.sum()
                if total_local_weight > 0:
                    local_weights = local_weights / total_local_weight
                total_global_weight = global_weights.sum()
                if total_global_weight > 0:
                    global_weights = global_weights / total_global_weight
                    
                # Combine the local and global weights using LOCAL_DOMINANCE.
                # The miners must perform well on both the global and local sets.
                weights = LOCAL_DOMINANCE * local_weights + (1 - LOCAL_DOMINANCE) * global_weights
    
                # This section calculates the "stale cliff" time, which is the global time at a specific block height.
                # The block height is determined by subtracting the product of BLOCKS_PER_EPOCH and EPOCH_CLIFF from the current epoch block.
                # Mathematically, this can be represented as:
                # stale_cliff_block = epoch_block - (BLOCKS_PER_EPOCH * EPOCH_CLIFF)
                # The global time at this block height is then queried from the blockchain.
                stale_cliff = subtensor.substrate.query(
                    module='Timestamp',
                    storage_function='Now',
                    block_hash = subtensor.get_block_hash( epoch_block - BLOCKS_PER_EPOCH * EPOCH_CLIFF )
                )
                # For each UID in the metagraph, the latest metadata is retrieved.
                # If the metadata is non-existent or its last modification time is before the stale cliff time,
                # the corresponding weight is set to zero, effectively ignoring stale or non-existent UIDs.
                for uid in metagraph.uids:
                    meta = get_latest_metadata( uid, metagraph, subtensor, CLIENT=CLIENT )
                    if meta == None or meta.last_modified < stale_cliff:
                        # Zero out non-existent or stale UIDS.
                        weights[ uid ] = 0
                        
                # Skew higher scores by temperature.
                weights = torch.exp(weights * TEMPERATURE)

                # Normalize the final weights to sum to 1.
                total_weight = weights.sum()
                if total_weight > 0:
                    weights = weights / total_weight

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
    parser.add_argument('--restart', action='store_true', help='Restart all evaluation history')

    # Add arguments from Bittensor modules for wallet and subtensor configurations.
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)

    # Parse the arguments to create a configuration object.
    config = bt.config(parser)

    # Set the chain endpoint for the subtensor (fixed value).
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'

    # Call the main function with the parsed configuration.
    main(config)
