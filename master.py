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
        run = wandb.init(project='cont', resume='allow', name=f'Master{my_uid}', config=config)
        
    # Load running hparams.
    hparams = load_hparams()
    # Optionally clear history on restart.
    master = LlamaForCausalLM( config = hparams.model_config ) 
    if config.restart:
        master = LlamaForCausalLM( config = hparams.model_config )
    else:
        master_metadata = get_latest_metadata( key = 'model', uid = my_uid, metagraph = metagraph, subtensor = subtensor, CLIENT = CLIENT )
        master = download_model( metadata = master_metadata, device='cpu', CLIENT=CLIENT ) 
    # Upload state to start.
    master_metadata = upload_model(
        key = 'model',
        wallet = wallet,
        model = master,
        block = int( time.time() ),
        extras = {},
        bucket = config.bucket,
        CLIENT = CLIENT,
    )  
    while True:
        try:
            # Sync the chain state.
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)            
            metagraph = subtensor.metagraph(netuid=config.netuid)       
            # Aggregate gradients from as many miners as we can until there is 3 pages. 
            start_block = subtensor.block
            # Aggregate for 5 blocks.
            while subtensor.block - start_block < 5: 
                print ('Wait for deltas...')
                deltas = []
                total_pages = 0
                for uid in metagraph.uids:
                    metadata = get_latest_metadata( key = 'delta', uid = uid, metagraph = metagraph, subtensor = subtensor, CLIENT = CLIENT )
                    if metadata == None: continue
                    deltas.append( metadata )
                    total_pages += metadata.n_pages
            if len(deltas) == 0:
                print ('No deltas to aggregate...')
                continue
                    
            # Now dowload the deltas and apply.
            print (f'Applying {len(deltas)} deltas ...')
            for delta_meta in deltas:
                try:
                    delta = download_model( metadata = delta_meta, device='cpu', CLIENT=CLIENT ) 
                    if delta == None: continue
                    print ('Apply Delta.')
                    for (name, master_param), (_, delta_param) in zip( master.named_parameters(), delta.named_parameters() ):
                        master_param.data.add_( metagraph.I[ delta_meta.uid ] * delta_param.data.to( master.device ) )
                except Exception as e:
                    # TODO remove the failed delta.
                    print (f'When applying deltas {e}')
                    continue
    
            # Delete previous master.
            if master_metadata != None:
                CLIENT.delete_object( Bucket = config.bucket, Key = master_metadata.filename )
                CLIENT.delete_object( Bucket = config.bucket, Key = master_metadata.metadata_filename )
            # Upload the new master
            master_metadata = upload_model(
                key = 'model',
                wallet = wallet,
                model = master,
                block = int( time.time() ),
                extras = { 'deltas': [ d.__dict__ for d in deltas ] },
                bucket = config.bucket,
                CLIENT = CLIENT,
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
