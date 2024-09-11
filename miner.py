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

import io
import os
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
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer


# Import common tooling.
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
    print('\n', '-' * 40, 'Config', '-' * 40,)
    print ( config )
    
    # Init Bittensor objects.
    wallet = bt.wallet( config = config )
    subtensor = bt.subtensor( config = config )
    metagraph = subtensor.metagraph( netuid = config.netuid )
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
    print('\n', '-' * 40, 'Objects', '-' * 40,)
    print ( f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}' )
    
    # Assert the chain commitment.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception: 
        subtensor.commit( wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket , '\n')
    
    # Init the model.
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( 'gpt2', verbose=False, clean_up_tokenization_spaces=True )
    tokenizer.pad_token = tokenizer.eos_token        
    model = LlamaForCausalLM( config = LlamaConfig(
        vocab_size = tokenizer.vocab_size,     
        hidden_size = 2040,   
        num_hidden_layers = 12,  
        num_attention_heads = 12,
        intermediate_size = 6144
    ))
    optimizer = optim.AdamW(
        model.parameters(),
        lr = config.learning_rate,  # Peak learning rate
        betas = ( config.optimizer_beta1, config.optimizer_beta2 ), # B1 and B2
        weight_decay = config.optimizer_weight_decay  # Weight decay
    )
    model.to(config.device)
    model.train()
                    
    # Init weights and biases
    run = None
    if config.use_wandb:
        name = f'M{wallet.hotkey.ss58_address[:5]}'
        run = wandb.init( project='bistro', resume = 'allow', name = name, config = config )
    
    # Main training loop.
    history = []
    while True:
        
        try:
            # Resync the chain state.
            subtensor = bt.subtensor( config = config )
            metagraph = subtensor.metagraph( netuid = config.netuid )
            
            # Pull my eval windo pages.
            eval_pages: Tuple[ str, int, str ] = SubsetFineWebEdu2Loader.next_pages( offset = subtensor.block, n_pages = config.eval_window, seed = my_uid )            
            dataset = SubsetFineWebEdu2Loader(
                batch_size = config.batch_size,
                sequence_length = 2048,
                pages_info = [ random.choice( eval_pages ) for _ in range(config.pages_per_step) ],
                tokenizer = tokenizer
            )
                
            # Train...
            for batch in dataset:
                
                # Forward pass
                input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                labels = input_ids.clone()
                labels = torch.where( labels == tokenizer.pad_token_id, -100, labels )
                outputs = model( input_ids = input_ids, labels = labels )
                
                # Accumulate gradients.
                outputs.loss.backward()
                if config.use_wandb: wandb.log({ "loss": outputs.loss.item() } )

                # Step the optimizer.
                optimizer.step()
                optimizer.zero_grad()
                    
            # Upload the delta to S3 and check state.
            history.append( upload_model(
                wallet = wallet,
                model = model,
                block = int(time.time()),
                extras = {},
                bucket = config.bucket,
                CLIENT = CLIENT,
            ))

        # Handle keyboard interrupts, stops training gracefully.
        except (KeyboardInterrupt, SystemExit):
            for el in history:
                print (f'Deleting: {el.filename}')
                CLIENT.delete_object( Bucket = config.bucket, Key = el.filename )
                CLIENT.delete_object( Bucket = config.bucket, Key = el.metadata_filename )
            break
        
        # Handle unknown exceptions, continue training after 5 seconds.
        except Exception as e:
            print (f"Error: {e}")
            time.sleep(5)
            continue

# Main function.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Miner script')
    parser.add_argument('--name', type=str, default=None, help='Optional miner name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network uid.')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--pages_per_step', type=int, default=1, help='Number of pages to eval the miner on every step.')
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.95, help='Beta2 for the optimizer')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.1, help='Weight decay for the optimizer')
    parser.add_argument('--eval_window', type=int, default=5, help='Number of pages to load')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    bt.wallet.add_args( parser )
    bt.subtensor.add_args( parser )
    config = bt.config( parser )   
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/' # Fix this value.
    main( config ) 