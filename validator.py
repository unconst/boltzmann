# The MIT License (MIT)
# Copyright © 2024 Chakana.tech

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
from transformers import AutoTokenizer
from typing import Dict, Optional
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

from common import upload_model, get_latest_metadata, download_model, hash_model
from dataset import SubsetFineWebEdu2Loader

# Instantiate my S3 client.
env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',
    aws_access_key_id = AWS_ACCESS_KEY_ID,
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY
)

# Main function.
def main( config ):
    print ( config )
    
    # Init Bittensor objects.
    wallet = bt.wallet( config = config )
    subtensor = bt.subtensor( config = config )
    metagraph = subtensor.metagraph( netuid = config.netuid )
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
    print ( f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}' )
    
    # Init weights and biases
    run = None
    if config.use_wandb:
        name = f'Validator-{wallet.hotkey.ss58_address[:5]}'
        run = wandb.init(project='bistro', resume = 'allow', name = name, config = config )
        
    # Remember delta for later removal.
    history = []
    weights = torch.zeros( (metagraph.n), dtype=torch.float32)
    while True:
        try:
            # Sync chain state.
            subtensor = bt.subtensor( config = config )
            metagraph = subtensor.metagraph( netuid = config.netuid )
                                
            # Get the next miner to eval.
            step_losses = torch.zeros( (metagraph.n), dtype=torch.float32)
            for uid in metagraph.uids:
            
                # Get the miner metadata
                miner_meta = get_latest_metadata( uid, metagraph, subtensor, CLIENT = CLIENT )
                if miner_meta == None:
                    # Miner meta is non existent or out of sync with the master.
                    continue

                # Download the delta.
                model = download_model( metadata = miner_meta, device = 'cpu', CLIENT = CLIENT )
                model.to(config.device)
                if model == None:
                    # Failed to download the delta.
                    continue
            
                # Pull pages from the miner windo.
                tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( 'gpt2', verbose=False, clean_up_tokenization_spaces=True )
                tokenizer.pad_token = tokenizer.eos_token        
                eval_pages: Tuple[ str, int, str ] = SubsetFineWebEdu2Loader.next_pages( offset = subtensor.block, n_pages = config.eval_window, seed = uid )
                dataset = SubsetFineWebEdu2Loader(
                    batch_size = config.batch_size,
                    sequence_length = 2048,
                    pages_info = [ random.choice( eval_pages ) for _ in range(config.pages_per_step) ],
                    tokenizer = tokenizer
                )
                                    
                # Eval the miner on the window.  
                losses = []      
                for batch in dataset:                
                    input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                    labels = input_ids.clone()
                    labels = torch.where( labels == tokenizer.pad_token_id, -100, labels )
                    with torch.no_grad():
                        outputs = model( input_ids=input_ids, labels=labels )
                    losses.append( outputs.loss.item() )                    
                    del input_ids, labels, outputs
                    torch.cuda.empty_cache()
                
                # Compute the loss.
                median_loss = np.mean( losses )
                step_losses[ uid ] = median_loss 
                print ( 'UID', uid, 'loss', loss  )
                if config.use_wandb: wandb.log({ "loss": loss } )
                
                # Remove the model.
                model.to('cpu')
                del model
                torch.cuda.empty_cache()
            
            # Compute weights.
            moving_average_losses = torch.zeros( (metagraph.n), dtype=torch.float32 )
            alpha = 0.1  # Smoothing factor for moving average
            for i, hist in enumerate(history):
                moving_average_losses = alpha * hist + (1 - alpha) * moving_average_losses
            print(f"Moving average losses: {moving_average_losses}")
                
            # Normalize the moving_average_losses
            moving_average_losses = moving_average_losses / moving_average_losses.sum()            
            exp_losses = torch.exp( config.temperature * normalized_losses)
            weights = exp_losses / exp_losses.sum()            
            print(f"Normalized weights: {weights}")
            if config.use_wandb:
                for i, weight in enumerate(weights):
                    wandb.log({ f"weight_{i}": weight.item() })
                    
                                
        # Handle keyboard interrupts, stops training gracefully.
        except (KeyboardInterrupt, SystemExit):
            if config.use_wandb and run != None: 
                api = wandb.Api()
                api_run = api.run(f"{run.entity}/{run.project}/{run.id}")
                api_run.delete()
            break
        
        # Handle unknown exceptions, continue training after 5 seconds.
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(5)
            continue

# Main function.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Miner script')
    parser.add_argument('--name', type=str, default=None, help='Optional name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network uid.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--eval_window', type=int, default=3, help='Number of pages to load')
    parser.add_argument('--pages_per_step', type=int, default=3, help='Number of pages to eval the miner on every step.')
    parser.add_argument('--temperature', type=int, default=3, help='How steep the exponentiation is.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    bt.wallet.add_args( parser )
    bt.subtensor.add_args( parser )
    config = bt.config( parser )   
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/' # Fix this value.
    main( config ) 