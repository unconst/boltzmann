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
         
    # Init empty weights.
    weights = torch.zeros( metagraph.S.shape, dtype = torch.float32 )
    while True:
        try:
            # Sync the chain state.
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)            
            metagraph = subtensor.metagraph(netuid=config.netuid)            
            
            # Get master metadata.
            print ('Get master metadata')
            master_uid = int(metagraph.S.argmax())
            master_meta = get_latest_metadata( key = 'model', uid = master_uid, metagraph = metagraph, subtensor = subtensor, CLIENT = CLIENT)
            if master_meta == None:
                print ('Waiting for master....')
                time.sleep(12)
                continue
            
            # Download the master.
            master = download_model( metadata = master_meta, device='cpu', CLIENT = CLIENT ) 
            if master == None:
                print ('Master was None, continue...')
                continue    
            
            # Make sure its on the right device.
            master.to( config.device ) 
            master.eval()
            master_hash = hash_model( master )
            
            # Get all delta metadatas.
            metas = []
            for uid in metagraph.uids:
                metadata = get_latest_metadata( key = 'delta', uid = uid, metagraph = metagraph, subtensor = subtensor, CLIENT = CLIENT )
                if metadata != None:
                    metas.append( metadata )
                    
            # Download all the deltas and apply them to the master.
            deltas = []
            for meta in metas:
                delta = download_model( metadata = meta, device='cpu', CLIENT=CLIENT ) 
                if delta != None: 
                    # Apply the delta to the master.
                    for (name, master_param), (_, delta_param) in zip( master.named_parameters(), delta.named_parameters()):
                        on_device = delta_param.data.to(master.device)
                        master_param.data.add_( metagraph.I[ meta.uid ] * on_device )
                        delta_param.data.to( 'cpu' )
                    deltas.append( (meta, delta) ) 
                    
            # Compute scores for each miner on their dataset sample.
            scores = torch.zeros_like( weights )
            for meta, delta in deltas:
                # Get a page for the uid for this delta.
                print (f'Evalling {meta.uid}')
                pages = [random.choice( SubsetFineWebEdu2Loader.next_pages(
                    offset = subtensor.block * hparams.window_speed,
                    n_pages = hparams.window_size,
                    seed = meta.uid 
                ))]
                dataset = SubsetFineWebEdu2Loader(
                    batch_size = config.batch_size,
                    sequence_length = hparams.sequence_length,
                    pages_info = pages,
                    tokenizer = hparams.tokenizer
                )
                # Remove the delta.
                for (name, master_param), (_, delta_param) in zip( master.named_parameters(), delta.named_parameters()):
                    on_device = delta_param.data.to(master.device)
                    master_param.data.sub_( metagraph.I[ meta.uid ] * on_device )
                    delta_param.data.to( 'cpu' )
                # Compute loss without the delta.
                print ('Running without delta...')
                total_loss_without_delta = 0
                for idx, batch in enumerate(dataset):
                    input_ids = torch.tensor(batch, dtype=torch.long).to(master.device)
                    labels = input_ids.clone()
                    labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                    with torch.no_grad():
                        outputs = master(input_ids=input_ids, labels=labels)
                        total_loss_without_delta += outputs.loss.item()
                        del input_ids, labels, outputs
                # Add the delta back.
                for (name, master_param), (_, delta_param) in zip( master.named_parameters(), delta.named_parameters()):
                    on_device = delta_param.data.to(master.device)
                    master_param.data.add_( metagraph.I[ meta.uid ] * on_device )
                    delta_param.data.to( 'cpu' )
                # Compute the loss with the delta.
                print ('Running with delta...')
                total_loss_with_delta = 0
                for idx, batch in enumerate(dataset):
                    input_ids = torch.tensor(batch, dtype=torch.long).to(master.device)
                    labels = input_ids.clone()
                    labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                    with torch.no_grad():
                        outputs = master(input_ids=input_ids, labels=labels)
                        total_loss_with_delta += outputs.loss.item()
                        del input_ids, labels, outputs
                # Compute scores as delta dif.
                torch.cuda.empty_cache() # Clean up.
                scores[ meta.uid ] = total_loss_without_delta - total_loss_with_delta
                
            # Set weights.
            print ( 'scores', scores )
            non_zero_scores = scores[scores != 0]
            non_zero_weights = torch.softmax(non_zero_scores, dim=0)
            next_weights = torch.zeros_like(scores)
            next_weights[scores != 0] = non_zero_weights
            next_weights = next_weights / next_weights.sum() if next_weights.sum() != 0 else next_weights
            weights = 0.9 * weights + ( 1 - 0.9 ) * next_weights
            print ( 'next_weights', next_weights.tolist() )              
            print ( 'weights', weights.tolist() )              
            subtensor.set_weights(
                wallet = wallet,
                netuid = metagraph.netuid,
                uids = metagraph.uids.tolist(),
                weights = weights.tolist(),
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )
                
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
