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
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from common import *
from dataset import SubsetFineWebEdu2Loader

# Instantiate the AWS S3 client.
env_config = {**dotenv_values(".env"), **os.environ}  # Load environment variables.
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')  # AWS access key ID.
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')  # AWS secret access key.
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',  # AWS region.
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
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
        
    # Load running hparams.
    hparams = load_hparams()
    # Load the evaluation history from S3 storage.
    history = load_history(my_uid, metagraph, subtensor, CLIENT)
    # Optionally clear history on restart.
    if config.restart:
        history = {}
        save_history(wallet, history, config.bucket, CLIENT)        
        master = LlamaForCausalLM( config = hparams.model_config )
        current_meta = upload_model(
            wallet = wallet,
            model = master,
            block = int(time.time()),  # Use current timestamp as block number.
            extras = {},  # Additional metadata can be added here.
            bucket = config.bucket,
            CLIENT = CLIENT,
        ) 

    # Remember the min moving global loss.
    upload_threshold = math.inf
    current_master_meta = None
    while True:
        try:
            # Sync the chain state.
            subtensor = bt.subtensor(config=config)            
            metagraph = subtensor.metagraph(netuid=config.netuid)
            
            # Sync the latest hparams from the global training state gist.
            hparams = load_hparams()
            
            # Get the master metadata.
            master_uid = int(metagraph.S.argmax())
            lastest_master_meta = get_latest_metadata( key = 'model', uid = master_uid, metagraph = metagraph, subtensor = subtensor, CLIENT = CLIENT)
            if lastest_master_meta == None:
                # Check if we are infact the master. Then upload our state.
                if master_uid == my_uid:
                    current_meta = upload_model(
                        wallet = wallet,
                        model = master,
                        block = int(time.time()),  # Use current timestamp as block number.
                        extras = {},  # Additional metadata can be added here.
                        bucket = config.bucket,
                        CLIENT = CLIENT,
                    )
                print ('No Valid master waiting ...')
                time.sleep(12)
                continue
                
            # If the master has changed or is None, download it.
            if current_master_meta == None or lastest_master_meta.model_hash != current_master_meta.model_hash:
                print ('Loading the new master...')
                current_master_meta = lastest_master_meta
                master = download_model( metadata = lastest_master_meta, device='cpu', CLIENT = CLIENT )
                master.to( config.device )
                master.eval()

            # Sample the next uid based on incentives.
            probabilities = metagraph.I + (hparams.base_probability / float(metagraph.n))
            probabilities /= probabilities.sum()
            next_uid = int(np.argmax(np.random.multinomial(1, probabilities)))

            # Retrieve the miner's metadata, which contains information about their delta.
            metadata = get_latest_metadata( key = 'delta', uid = next_uid, metagraph = metagraph, subtensor = subtensor, CLIENT=CLIENT)
            if metadata is None:
                # Start again.
                continue 
            
            # Download the miner delta based on the metadata.
            delta = download_model( metadata = metadata, device='cpu', CLIENT=CLIENT )
            if delta is None:
                # Start again.
                continue
            
            # Apply the delta to the master model.
            for (name, master_param), (_, delta_param) in zip( master.named_parameters(), delta.named_parameters() ):
                master_param.data.add_( delta_param.data.to( master.device ) )
                
            # Select local and global pages to eval the model on.
            local_losses = []
            global_losses = []
            local_page = random.choice(SubsetFineWebEdu2Loader.next_pages(
                offset = subtensor.block * hparams.window_speed,
                n_pages = hparams.window_size,
                seed = next_uid
            ))
            local_dataset = SubsetFineWebEdu2Loader(
                batch_size = config.batch_size,
                sequence_length = hparams.sequence_length,
                pages_info = [local_page],
                tokenizer = hparams.tokenizer
            )
            global_dataset = SubsetFineWebEdu2Loader(
                batch_size = config.batch_size,
                sequence_length = hparams.sequence_length,
                num_pages = 1,
                tokenizer = hparams.tokenizer
            )
            global_page = global_dataset.pages[0]
            
            # Evaluate the model on the local dataset.
            for batch in local_dataset:
                # Convert the batch to tensors and move to the device.
                input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                labels = input_ids.clone()  # Clone input_ids for labels.
                # Mask the padding tokens.
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                with torch.no_grad():
                    # Forward pass through the model with scaling.
                    with torch.amp.autocast(config.device, dtype=torch.bfloat16):
                        outputs = master(input_ids=input_ids, labels=labels)
                # Append the loss to the list.
                local_losses.append(outputs.loss.item())
                # Clean up to free memory.
                del input_ids, labels, outputs
                torch.cuda.empty_cache()

            # Evaluate the model on the global dataset.
            for batch in global_dataset:
                input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                with torch.no_grad():
                    # Forward pass through the model with scaling.
                    with torch.amp.autocast(config.device, dtype=torch.bfloat16):
                        outputs = master(input_ids=input_ids, labels=labels)
                global_losses.append(outputs.loss.item())
                del input_ids, labels, outputs
                torch.cuda.empty_cache()

            # Record the evaluation event.
            # Create an event dictionary with all relevant information.
            event = {
                'block': int(subtensor.block),
                'next_uid': int(next_uid),
                'local_page': int(local_page[1]),
                'global_page': int(global_page[1]),
                'local_loss': float(np.mean(local_losses)),
                'global_loss': float(np.mean(global_losses)),
                'local_losses': [float(v) for v in local_losses],
                'global_losses': [float(v) for v in global_losses],
            }
            print(
                f'UID {next_uid}, Block {subtensor.block}, '
                f'Local Page {int(local_page[1])}, Local Loss {float(np.mean(local_losses))}, '
                f'Global Page {int(global_page[1])}, Global Loss {float(np.mean(global_losses))}'
            )
            # Append the event to the history.
            if next_uid not in history: history[next_uid] = [] 
            history[next_uid].append(event)
            if config.use_wandb:
                # Log the event to Weights and Biases.
                wandb.log(event)
                
            # Save the eval history to the chain and then reload.
            save_history( wallet, history, config.bucket, CLIENT )

            # Initialize tensors for local and global weights.
            local_loss = torch.zeros(metagraph.uids.shape)
            global_loss = torch.zeros(metagraph.uids.shape)

            # Compute the moving average local and global losses.
            for uid, samples in history.items():
                last_block, moving_global_loss, moving_local_loss = None, 0, 0
                for sample in samples:
                    next_local_loss = float(np.mean(sample['local_losses']))
                    next_global_loss = float(np.mean(sample['global_losses']))
                    if np.isnan(next_local_loss) or np.isnan(next_global_loss):
                        continue # Skip NaN values.
                    block = int(sample['block'])
                    alpha = hparams.base_alpha * (block - last_block) if last_block is not None else 1
                    moving_global_loss = alpha * next_global_loss + (1 - alpha) * moving_global_loss
                    moving_local_loss = alpha * next_local_loss + (1 - alpha) * moving_local_loss
                    last_block = block
                local_loss[int(uid)] = moving_local_loss
                global_loss[int(uid)] = moving_global_loss
            print('local_loss:', local_loss.tolist())
            print('global_loss:', global_loss.tolist())
                
            # Compute scoring for local loss.
            local_weights = torch.zeros(metagraph.uids.shape)
            if local_loss.sum() > 0:
                local_weights = -(local_loss / local_loss.sum())
                non_zero = local_loss.nonzero(as_tuple=True)
                local_min = local_loss[ non_zero ].min()
                local_max = local_loss[ non_zero ].max()
                local_weights[ non_zero ] = (local_loss[ non_zero ] - local_min) / (local_max - local_min + 1e-10)
            print('local_weights:', local_weights.tolist())
  
            # Compute scoring for global loss.
            global_weights = torch.zeros(metagraph.uids.shape)
            if global_loss.sum() > 0:
                global_weights = -(global_loss / global_loss.sum())
                non_zero = global_loss.nonzero(as_tuple=True)
                global_min = global_loss[ non_zero ].min()
                global_max = global_loss[ non_zero ].max()
                global_weights[ non_zero ] = (global_loss[ non_zero ] - global_min) / (global_max - global_min + 1e-10)
            print('global_weights:', global_weights.tolist())
                                
            # Combine local and global loss scoring.
            weights = hparams.local_dominance * local_weights + (1 - hparams.local_dominance) * global_weights                    
            weights = torch.exp(weights * hparams.temperature)
            weights = weights / weights.sum() if weights.sum() != 0 else weights
            print('Weights:', weights.tolist())

            # Set the computed weights on the chain using the wallet.
            subtensor.set_weights(
                wallet=wallet,
                netuid=metagraph.netuid,
                uids=metagraph.uids.tolist(),
                weights=weights.tolist(),
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )            
            
            # If we are the master validator, check if the latest model has beaten the 
            # threshold for upload.
            uid_score = global_loss[next_uid]
            if config.use_wandb: wandb.log({ 'uid_score': uid_score, 'upload_threshold': upload_threshold  })
            if uid_score < upload_threshold and my_uid == master_uid:
                print ('New Master, uploading state.')
                upload_threshold = uid_score * hparams.epsilon
                CLIENT.delete_object( Bucket=config.bucket, Key=current_master_meta.filename )
                CLIENT.delete_object( Bucket=config.bucket, Key=current_master_meta.metadata_filename )
                current_master_meta = upload_model(
                    key = 'model',
                    wallet = wallet,
                    model = master,
                    block = int(time.time()),
                    extras = { 'delta': metadata.__dict__ }, # Record the delta we just applied.
                    bucket = config.bucket,
                    CLIENT = CLIENT,
                ) 
                
            # Otherwise, simply remove the delta and continue
            else:
                for (name, master_param), (_, delta_param) in zip( master.named_parameters(), delta.named_parameters() ):
                    master_param.data.sub_( delta_param.data.to( master.device ) )

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
