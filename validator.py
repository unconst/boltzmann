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
         
    # Init empty weights.
    last_master_sync = 0
    upload_history = []
    weights = torch.zeros( metagraph.S.shape, dtype = torch.float32 )
    while True:
        try:
            # Sync the chain state.
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)            
            metagraph = subtensor.metagraph(netuid=config.netuid)  
            
            # Sync the full model state if we have gone further than the epoch.
            if subtensor.block - last_master_sync > hparams.epoch_length:
                print ('Resyncing full training state.')
                try:
                    master_uid = int(metagraph.S.argmax())
                    master_meta = get_latest_metadata( key = 'model', uid = master_uid, metagraph = metagraph, subtensor = subtensor )
                    master = download_model( metadata = master_meta, device='cpu', CLIENT = CLIENT )
                    if master == None:
                        raise ValueError('No Master...') 
                except Exception as e:
                    # Upload the new master
                    master = LlamaForCausalLM( config = hparams.model_config ) 
                    upload_history.append( upload_model(
                        key = 'model',
                        wallet = wallet,
                        model = master,
                        block = subtensor.block,
                        extras = {},
                        bucket = config.bucket,
                        CLIENT = CLIENT,
                    ) )
                    time.sleep(12)
                    continue
                last_master_sync = subtensor.block    
            
            # Get block.
            block = subtensor.block
            step_size = hparams.blocks_per_step
            next_sync_block = (int(subtensor.block / step_size) * step_size) + step_size
            while True:
                block = subtensor.block
                if block >= next_sync_block:
                    break
                print (f'Waiting for sync block: {next_sync_block} current: {block}')
                time.sleep(4)
                continue
                             
            # Get the mask for the sync block.
            mask_indices = {}
            compression_factor = hparams.compression
            print(f'Creating Mask with compression: {compression_factor} for block: {next_sync_block}')
            # We seed the mask from the block height.
            np.random.seed( next_sync_block ) 
            for name, param in master.named_parameters():
                next_mask = torch.from_numpy(np.random.rand(*param.shape) < (1 / compression_factor)).float()
                indices = next_mask.nonzero(as_tuple=False).flatten()
                mask_indices[ name ] = indices
                
            # Sync and average all the masks from peers on the sync block.
            print ('Downloading the masks')
            mask_count = 0
            masks_dicts_values = {}
            for uid in metagraph.uids:
                print (f'Getting metadata from uid: {uid}')
                metadata = get_metadata_for_block( 
                    key = 'mask', 
                    uid = uid, 
                    block = next_sync_block,
                    metagraph = metagraph, 
                    subtensor = subtensor,
                )
                if metadata == None: 
                    print (f'No metadata from uid: {uid}')
                    continue
                # Download the compressed state_dict.
                print (f'Downloading mask from uid: {uid}')
                mask = download_model( metadata = metadata, device='cpu', CLIENT=CLIENT, state_dict = True )
                if mask == None: 
                    print (f'No mask from uid: {uid}')
                    continue
                print (f'Unpacking mask from uid: {uid}')
                mask_count += 1
                for name in mask.keys():
                    param_shape = master.get_parameter(name).shape
                    mask_values = mask[name]['values']
                    indices = mask_indices[name] 
                    decompressed = torch.zeros(param_shape, device='cpu').flatten() 
                    decompressed[indices] = mask_values
                    if name not in masks_dicts_values:
                        masks_dicts_values[name] = decompressed.view(param_shape)
                    else:
                        masks_dicts_values[name] += decompressed.view(param_shape)
                        
            if mask_count == 0:
                print ('No masks to merge. Continuing...')
                continue

            # Average the mask values
            print (f'Averaging {mask_count} masks.')
            for key in masks_dicts_values:
                masks_dicts_values[key] /= mask_count
                
            # Set these values into the model
            print (f'Applying average of {mask_count} masks to the master.')
            for name, param in master.named_parameters():
                if name in masks_dicts_values:
                    with torch.no_grad():
                        param.copy_(masks_dicts_values[name])
                        
            # Uploading the new state.
            upload_history.append( upload_model(
                key = 'model',
                wallet = wallet,
                model = master,
                block = next_sync_block,
                extras = {},
                bucket = config.bucket,
                CLIENT = CLIENT,
            ) )
            # Delete history over allowed.
            if len(upload_history) > 10:
                to_delete = upload_history.pop(0)
                CLIENT.delete_object( Bucket=config.bucket, Key=to_delete.filename )
                CLIENT.delete_object( Bucket=config.bucket, Key=to_delete.metadata_filename )

                
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
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--restart', action='store_true', help='Restart all evaluation history')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
    main(config)



# # Compute scores for each miner on their dataset sample.
#             scores = torch.zeros_like( weights )
#             for meta, delta in deltas:
#                 # Get a page for the uid for this delta.
#                 print (f'Evalling {meta.uid}')
#                 pages = [random.choice( SubsetFineWebEdu2Loader.next_pages(
#                     offset = subtensor.block * hparams.window_speed,
#                     n_pages = hparams.window_size,
#                     seed = meta.uid 
#                 ))]
#                 dataset = SubsetFineWebEdu2Loader(
#                     batch_size = config.batch_size,
#                     sequence_length = hparams.sequence_length,
#                     pages_info = pages,
#                     tokenizer = hparams.tokenizer
#                 )
#                 # Remove the delta.
#                 for (name, master_param), (_, delta_param) in zip( master.named_parameters(), delta.named_parameters()):
#                     on_device = delta_param.data.to(master.device)
#                     master_param.data.sub_( metagraph.I[ meta.uid ] * on_device )
#                     delta_param.data.to( 'cpu' )
#                 # Compute loss without the delta.
#                 print ('Running without delta...')
#                 total_loss_without_delta = 0
#                 for idx, batch in enumerate(dataset):
#                     input_ids = torch.tensor(batch, dtype=torch.long).to(master.device)
#                     labels = input_ids.clone()
#                     labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
#                     with torch.no_grad():
#                         outputs = master(input_ids=input_ids, labels=labels)
#                         total_loss_without_delta += outputs.loss.item()
#                         del input_ids, labels, outputs
#                 # Add the delta back.
#                 for (name, master_param), (_, delta_param) in zip( master.named_parameters(), delta.named_parameters()):
#                     on_device = delta_param.data.to(master.device)
#                     master_param.data.add_( metagraph.I[ meta.uid ] * on_device )
#                     delta_param.data.to( 'cpu' )
#                 # Compute the loss with the delta.
#                 print ('Running with delta...')
#                 total_loss_with_delta = 0
#                 for idx, batch in enumerate(dataset):
#                     input_ids = torch.tensor(batch, dtype=torch.long).to(master.device)
#                     labels = input_ids.clone()
#                     labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
#                     with torch.no_grad():
#                         outputs = master(input_ids=input_ids, labels=labels)
#                         total_loss_with_delta += outputs.loss.item()
#                         del input_ids, labels, outputs
#                 # Compute scores as delta dif.
#                 torch.cuda.empty_cache() # Clean up.
#                 scores[ meta.uid ] = total_loss_without_delta - total_loss_with_delta
                
#             # Set weights.
#             print ( 'scores', scores.tolist() )
#             non_zero_scores = scores[scores != 0]
#             non_zero_weights = torch.softmax(non_zero_scores, dim=0)
#             next_weights = torch.zeros_like(scores)
#             next_weights[scores != 0] = non_zero_weights
#             next_weights = next_weights / next_weights.sum() if next_weights.sum() != 0 else next_weights
#             weights = 0.9 * weights + ( 1 - 0.9 ) * next_weights
#             print ( 'next_weights', next_weights.tolist() )              
#             print ( 'weights', weights.tolist() )              
#             subtensor.set_weights(
#                 wallet = wallet,
#                 netuid = metagraph.netuid,
#                 uids = metagraph.uids.tolist(),
#                 weights = weights.tolist(),
#                 wait_for_inclusion=False,
#                 wait_for_finalization=False,
#             )