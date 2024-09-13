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
    print('Bucket:', config.bucket)
    
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
    print ('Optimizer', optimizer )
    print ('Model', 'llama', '\n')
    model.to(config.device)
    model.train()
                    
    # Init weights and biases
    if config.use_wandb:
        run = wandb.init( project='cont', resume = 'allow', name = f'M{my_uid}', config = config )
    
    # Main training loop.
    n_epochs = 0
    current_meta = None 
    while True:
        
        try:
            # Resync the chain state.
            n_epochs += 1
            print('\n', '=' * 40, f'Epoch: {n_epochs}', '=' * 40, '\n')
            subtensor = bt.subtensor( config = config )
            metagraph = subtensor.metagraph( netuid = config.netuid )
            if config.use_wandb: wandb.log({ f"Incentive({my_uid})": float(metagraph.I[ my_uid ]) } )

            # Iterate pages per epoch training on the next from my window.
            for step in range(config.pages_per_epoch):
                
                # Get the current window.
                eval_pages: Tuple[ str, int, str ] = SubsetFineWebEdu2Loader.next_pages( 
                    offset = subtensor.block * config.window_speed + 100, # Sampling from the future.
                    n_pages = config.window_size, 
                    seed = my_uid 
                )
                
                # Pull a random page from my eval window.
                dataset = SubsetFineWebEdu2Loader(
                    batch_size = config.batch_size,
                    sequence_length = 2048,
                    pages_info = [ random.choice( eval_pages ) ],
                    tokenizer = tokenizer
                )
                    
                # Train on the epoch.
                for idx, batch in enumerate(dataset):
                    
                    # Forward pass
                    input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                    labels = input_ids.clone()
                    labels = torch.where( labels == tokenizer.pad_token_id, -100, labels )
                    outputs = model( input_ids = input_ids, labels = labels )
                    
                    # Accumulate gradients.
                    outputs.loss.backward()
                    print ( "epoch:", n_epochs, "step:", f"{step}/{config.pages_per_epoch}", "batch:", f"{idx}", "loss:", outputs.loss.item() )
                    if config.use_wandb: wandb.log({ "epoch": n_epochs, "step": step, "batch": idx, "loss": outputs.loss.item() } )

                    # Step the optimizer.
                    optimizer.step()
                    optimizer.zero_grad()
                    
            # Delete the previous model.
            if current_meta != None:
                CLIENT.delete_object( Bucket = config.bucket, Key = current_meta.filename )
                CLIENT.delete_object( Bucket = config.bucket, Key = current_meta.metadata_filename )
                
            # Upload the next to S3.
            current_meta = upload_model(
                wallet = wallet,
                model = model,
                block = int(time.time()),
                extras = {},
                bucket = config.bucket,
                CLIENT = CLIENT,
            )

        # Handle keyboard interrupts, stops training gracefully.
        except (KeyboardInterrupt, SystemExit):
            if current_meta != None:
                CLIENT.delete_object( Bucket = config.bucket, Key = current_meta.filename )
                CLIENT.delete_object( Bucket = config.bucket, Key = current_meta.metadata_filename )
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
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.95, help='Beta2 for the optimizer')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.1, help='Weight decay for the optimizer')
    parser.add_argument('--window_size', type=int, default=5, help='Size of eval window used to evaluate the miner')
    parser.add_argument('--window_speed', type=int, default=5, help='Speed that eval window moves forward across series.')
    parser.add_argument('--pages_per_epoch', type=int, default=5, help='Pages to train per epoch.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    bt.wallet.add_args( parser )
    bt.subtensor.add_args( parser )
    config = bt.config( parser )   
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/' # Fix this value.
    main( config ) 