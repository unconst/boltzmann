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
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
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
        
    # Load the model from bucket if exists.
    hparams = load_hparams()
    upload_history = []
    model = LlamaForCausalLM(config=hparams.model_config) 
    if not config.restart: 
        try:
            master_filename = f'master-{wallet.hotkey.ss58_address}.pt'
            unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
            CLIENT.download_file(config.bucket, master_filename, unique_temp_file)
            master_state_dict = torch.load(unique_temp_file, map_location='cpu', weights_only=True)
            model.load_state_dict(master_state_dict)
            upload_history.append(master_filename)
        except Exception as e:
            raise ValueError("There is no master to continue from. Run with --restart")
    model.to(config.device)  # TODO: Ensure 'device' is defined in config
        
    # Start.
    last_mask_sync = 0
    while True:
        try:
            print('Loading chain state:')
            start_time = time.time()
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
            print(f'Loading chain state completed in {time.time() - start_time} seconds') 
            
            print('Getting blocks to sync:')
            start_time = time.time() 
            block = subtensor.block
            all_sync_blocks = [last_mask_sync + i + 1 for i in range(block - last_mask_sync)]
            last_mask_sync = block
            print(f'Getting blocks to sync completed in {time.time() - start_time} seconds')  # Print timing after this step
        
            # Get the mask for the sync block.
            print(f'Downloading masks for blocks: {all_sync_blocks}')  # Start timing
            full_sync_start_time = time.time()
            for blk in all_sync_blocks:
                
                print(f'Getting filenames for blk: {blk}...')
                start_time = time.time()
                if 'buckets' not in locals():
                    buckets = []
                    for uid in metagraph.uids:
                        buckets.append(subtensor.get_commitment(config.netuid, uid))
                mask_filenames = []
                for uid in metagraph.uids:
                    mask_filenames.append(f"mask-{str(metagraph.hotkeys[uid])}-{blk}.pt")
                print(f'Get filenames completed in {time.time() - start_time} seconds')
            
                print(f'Downloading mask for blk: {blk}:')
                start_time = time.time()
                temp_files = []
                n_downloaded = 0
                def download_file(bucket, filename):
                    try:
                        unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                        CLIENT.download_file(bucket, filename, unique_temp_file)
                        return unique_temp_file
                    except:
                        return None
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(download_file, bucket, filename) for bucket, filename in zip(buckets, mask_filenames)]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            temp_files.append(result)
                            n_downloaded += 1
                print(f'Downloading {n_downloaded} masks completed in {time.time() - start_time} seconds')
                
                # Break the loop when there is nothing to download.
                if n_downloaded == 0:
                    continue
                
                print(f'Creating sync mask for block: {blk}')
                mask_indices = {}
                torch.manual_seed(blk)
                start_time = time.time()
                for name, param in model.named_parameters():
                    param = param.to(config.device)
                    next_mask = (torch.rand(param.shape, device=config.device) < (1 / hparams.compression)).float()
                    indices = next_mask.flatten().nonzero(as_tuple=False).flatten()
                    mask_indices[name] = indices
                print(f'Creating sync block mask completed in {time.time() - start_time} seconds')  # Print timing after this step
            
                # Loading state dicts
                print(f'Loading state dicts for block: {blk}:')
                start_time = time.time()
                mask_count = 0
                masks_dicts_values = {}
                for file in temp_files:
                    mask = torch.load(file, map_location='cpu', weights_only=True)
                    mask_count += 1
                    for name in mask.keys():
                        mask_values = mask[name]['values']
                        if torch.isnan(mask_values).any():
                            continue
                        param_shape = model.get_parameter(name).shape
                        indices = mask_indices[name] 
                        decompressed = torch.zeros(param_shape, device='cpu').flatten() 
                        decompressed[indices] = mask_values
                        if name not in masks_dicts_values:
                            masks_dicts_values[name] = decompressed.view(param_shape)
                        else:
                            masks_dicts_values[name] += decompressed.view(param_shape)
                print(f'Loading state dicts completed in {time.time() - start_time} seconds')
                
                # Average the mask values
                print(f'Averaging {mask_count} masks for block: {blk}')
                start_time = time.time()
                for key in masks_dicts_values.keys():
                    masks_dicts_values[key] /= mask_count
                print(f'Averaged state dicts in {time.time() - start_time} seconds')
                
                # Set these values into the model
                print(f'Applying {mask_count} masks for block: {blk}:')
                start_time = time.time()  # Start timing
                for name, param in model.named_parameters():
                    indices = mask_indices[name]
                    if name in masks_dicts_values:
                        if masks_dicts_values[name].shape == param.shape:
                            # Apply the mask values to the flattened param data.
                            on_device = masks_dicts_values[name].to(model.device).flatten()
                            param_flat = param.data.flatten()
                            param_flat[indices] = on_device[indices]
                            param.data.copy_(param_flat.view(param.shape))
                            del on_device, param_flat
                        else:
                            print(f"Shape mismatch for {name}: expected {param.shape}, got {masks_dicts_values[name].shape}")
                del masks_dicts_values
                print(f'Applying {mask_count} masks completed in {time.time() - start_time} seconds')  # Print timing after this step
                
                # Delete files.
                print(f'Deleting files for block: {blk}.')
                start_time = time.time()
                for file in temp_files:
                    os.remove(file)
                print(f'Deleting files completed in {time.time() - start_time} seconds')
                
            # Print completion
            torch.cuda.empty_cache()
            print(f'Downloading masks for blocks: {all_sync_blocks} in {time.time() - full_sync_start_time} seconds')

            # Upload the masked weights.
            print('Uploading master:')
            start_time = time.time()
            model_state_dict = model.state_dict()
            upload_filename = f'master-{wallet.hotkey.ss58_address}.pt'
            with io.BytesIO() as module_buffer:
                torch.save(model_state_dict, module_buffer)
                module_buffer.seek(0)  # Reset the buffer's position to the beginning.
                CLIENT.upload_fileobj(module_buffer, config.bucket, upload_filename)
            CLIENT.put_object_acl(
                Bucket=config.bucket,
                Key=upload_filename,
                GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
                GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
            )
            upload_history.append(upload_filename)
            print(f'Uploading master completed in {time.time() - start_time} seconds')

            print('Deleting history:')
            start_time = time.time()
            if len(upload_history) > 5:
                CLIENT.delete_object(Bucket=config.bucket, Key=upload_history.pop(0))
            print(f'Deleting history completed in {time.time() - start_time} seconds')
                 
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
    parser = argparse.ArgumentParser(description='Validator script')
    parser.add_argument('--name', type=str, default=None, help='Optional name')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network uid.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--restart', action='store_true', help='Restart all evaluation history')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
    main(config)
