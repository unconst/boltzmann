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
from transformers import AutoTokenizer
from typing import Dict, Optional
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

from common import get_latest_metadata, download_model, load_history, save_history
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
    print('\n', '=' * 40, 'Config', '=' * 40,)
    print ( config )
    
    # Init Bittensor objects.
    wallet = bt.wallet( config = config )
    subtensor = bt.subtensor( config = config )
    metagraph = subtensor.metagraph( netuid = config.netuid )
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
    print('\n', '=' * 40, 'Objects', '=' * 40,)
    print ( f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}' )
    
    # Assert the chain commitment.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception: 
        subtensor.commit( wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket)
    
    # Init weights and biases
    if config.use_wandb:
        run = wandb.init(project='cont', resume = 'allow', name = f'V{my_uid}', config = config )
        
    # Remember delta for later removal.
    n_uids = 0
    n_epochs = 0
    n_samples = 0
    # Load the history from s3.
    history = load_history( my_uid, metagraph, subtensor, CLIENT )
            
    while True:
        try:
            
            # Sync the chain state and reconnect to subtensor.
            n_epochs += 1
            subtensor = bt.subtensor( config = config )
                                                        
            # Compute the epoch length from blocks_per_uid and uids_per_epoch.
            # Attains the block of the last epoch start with epoch_length num blocks between them. 
            epoch_length = config.blocks_per_uid * config.uids_per_epoch # Get the epoch length in blocks.
            epoch_block = int( (subtensor.block / epoch_length) * epoch_length) # Get the block at the last epoch.
            epoch_hash = subtensor.get_block_hash( epoch_block ) # Get the hash of the block at the last epoch.
            metagraph = subtensor.metagraph( netuid = config.netuid, block = epoch_block ) # Sync the graph at the epoch block.
            
            # Compute the epoch series function which returns a UID per block during the epoch. For instance:
            # uids_per_epoch = 3
            # blocks_per_uid = 2
            # epoch_uids = [ 1, 4, 2 ]
            # uid_for_block( [ 0, 1, 2, 3, 4, 5 ] )  = [ 1, 1, 4, 4, 2, 2] 
            # We sample the UIDs based on the incentive of each miner at the epoch_block to miners with higher incentive are sampled more often.
            np.random.seed( int( epoch_hash[:10], 16 ) ) # Seed numpy randomness from the block hash
            probabilities = ( metagraph.I + ( 0.001 / float(metagraph.n) ) ) / (( metagraph.I + ( 0.001 / float(metagraph.n) ) ).sum()) # Get probabilities from the metagraph.
            epoch_uids = np.random.choice( len( probabilities ), size = config.uids_per_epoch, replace = config.uids_per_epoch > metagraph.n, p = probabilities ) # Get the UIDs to sample for this epoch.
            np.random.shuffle( epoch_uids ) # Shuffle the uids over the epoch
            epoch_uids = [ int( u ) for u in epoch_uids ] # convert to integer list.
            def uid_for_block( block: int ) -> int:
                current_epoch_index = config.uids_per_epoch * ( (block % ( config.uids_per_epoch * config.blocks_per_uid ) ) + 1 ) / (config.uids_per_epoch * config.blocks_per_uid)  
                return int( epoch_uids[ min( len(epoch_uids) - 1 , int( current_epoch_index ) )] )
            
            # Print epoch information.
            print('\n', '=' * 40, f'Epoch: {n_epochs}', '=' * 40, '\n')
            print ( 'uids_per_epoch:', config.uids_per_epoch  )
            print ( 'blocks_per_uid:', config.blocks_per_uid )
            print ( 'epoch_length:', epoch_length )
            print ( 'epoch_block:', epoch_block )
            print ( 'epoch_hash:', epoch_hash )
            print ( 'probabilities:', probabilities )
            print ( 'epoch_uids:', epoch_uids )
            
            # Iterate over each UID in the series.
            for index in range( config.uids_per_epoch ):
                
                # Here we get the current UID to eval at this block. 
                n_uids += 1
                current_uid = uid_for_block( subtensor.block )
                print('\n\n', '-' * 20, f'current_uid: { current_uid }', '-' * 20, '\n')
                
                # Here we get the miner metadata at this current block which tells us about their model.
                miner_meta = get_latest_metadata( current_uid, metagraph, subtensor, CLIENT = CLIENT )
                if miner_meta == None:
                    # Wait until we are evaluating the next miner
                    print ('No valid metadata for uid.')
                    print ('Waiting for next valid uid ...')
                    while uid_for_block( subtensor.block ) == current_uid:
                        time.sleep( 12 )
                    continue
                
                # Here we download the miner model at this block. If the download fails or is None
                # we will wait until the next uid to evaluate by stepping blocks.
                try:
                    model = download_model( metadata = miner_meta, device = 'cpu', CLIENT = CLIENT )
                    if model == None:
                        raise ValueError('Miner model is Non-existent.')
                except Exception as e:
                    print ( 'e:', e )
                    # Wait until we are evaluating the next miner uid.
                    print ('Waiting for next valid uid ...')
                    while uid_for_block( subtensor.block ) == current_uid:
                        time.sleep( 12 )
                    continue
                model.to(config.device)
                
                # Load the tokenizer.
                tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( 'gpt2', verbose=False, clean_up_tokenization_spaces=True )
                tokenizer.pad_token = tokenizer.eos_token
                
                # Get the eval pages based on the current epoch block (these blocks are known to miners.)
                # The next pages function returns a unique set of pages for each miner based on the epoch block.
                # All validators see the same window of pages for each miner.
                eval_pages: List[ Tuple[ str, int, str ] ] = SubsetFineWebEdu2Loader.next_pages( 
                    offset = epoch_block * config.window_speed, 
                    n_pages = config.window_size, 
                    seed = current_uid 
                )
                # Get the miner holdout pages. This is a random sample from outside the 
                # eval window of a miner (although there may be some overlap). The window is seeded 
                # by the hash of the epoch block which ensure that the miners can't know this until the epoch
                # block passes, but all validators will see the same set.
                holdout_pages: List[ Tuple[ str, int, str ] ] = SubsetFineWebEdu2Loader.next_pages( 
                    offset = epoch_block * config.window_speed, 
                    n_pages = config.window_size, 
                    seed = epoch_hash 
                )    
        
                # We continously pull the current miner to eval until it changes. 
                # When it changes we will remove the model and restart the loop.
                while uid_for_block( subtensor.block ) == current_uid:
                    
                    # Select random pages to eval on.
                    current_block = subtensor.block
                    eval_page = random.choice( eval_pages )
                    holdout_page = random.choice( holdout_pages )
                    eval_dataset = SubsetFineWebEdu2Loader(
                        batch_size = config.batch_size,
                        sequence_length = 2048,
                        pages_info = [ eval_page ],
                        tokenizer = tokenizer
                    )
                    holdout_dataset = SubsetFineWebEdu2Loader(
                        batch_size = config.batch_size,
                        sequence_length = 2048,
                        pages_info = [ holdout_page ],
                        tokenizer = tokenizer
                    )
                                        
                    # Compute losses for miner model on the eval set.  
                    eval_losses = []      
                    for batch in eval_dataset:                
                        input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                        labels = input_ids.clone()
                        labels = torch.where( labels == tokenizer.pad_token_id, -100, labels )
                        with torch.no_grad():
                            outputs = model( input_ids=input_ids, labels=labels )
                        eval_losses.append( outputs.loss.item() )                    
                        del input_ids, labels, outputs
                        torch.cuda.empty_cache()
                    
                    # Compute losses for miner model on the holdout set.  
                    holdout_losses = []      
                    for batch in holdout_dataset:                
                        input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                        labels = input_ids.clone()
                        labels = torch.where( labels == tokenizer.pad_token_id, -100, labels )
                        with torch.no_grad():
                            outputs = model( input_ids=input_ids, labels=labels )
                        holdout_losses.append( outputs.loss.item() )                    
                        del input_ids, labels, outputs
                        torch.cuda.empty_cache()
                    
                    # Record the event.
                    n_samples += 1
                    if current_uid not in history: history[ current_uid ] = []
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
                    history[ current_uid ].append( event )
                    if config.use_wandb: wandb.log( event )
                    
                # Save the history to s3.
                # Only save at the end of the step.
                save_history( wallet, history, config.bucket, CLIENT )

                # Remove the model here and start again.
                model.to('cpu')
                del model
                torch.cuda.empty_cache()
                
            ###############
            ## End Epoch ##
            ###############
                           
            # Compute the minimum model moving average.
            evals = load_history( my_uid, metagraph, subtensor, CLIENT )
            eval_weights = torch.zeros( metagraph.uids.shape )
            hold_weights = torch.zeros( metagraph.uids.shape )
            
            # Compute the moving average of the eval loss and the holdout loss
            # for each uid.
            for uid in evals.keys():
                last_block = None
                moving_hold_loss = 0
                moving_eval_loss = 0
                for sample in evals[uid]:
                    block = int(sample['block'])
                    next_eval_loss = float(np.mean(sample['eval_losses']))
                    next_hold_loss = float(np.mean(sample['holdout_losses']))
                    alpha = base_alpha * (block - last_block) if last_block != None else 1
                    moving_hold_loss = alpha * next_hold_loss + (1 - alpha) * moving_hold_loss
                    moving_eval_loss = alpha * next_eval_loss + (1 - alpha) * moving_eval_loss
                    last_block = block
                eval_weights[ int(uid) ] = moving_eval_loss
                hold_weights[ int(uid) ] = moving_hold_loss
                
            # Normalize the non-zero values on the range (0,1) with the highest value at 1 and the smallest value at 0
            # The combine the eval and holdout losses to attain the absolute weights update.
            non_zero_eval_indices = eval_weights.nonzero()
            non_zero_hold_indices = hold_weights.nonzero()
            eval_weights[non_zero_indices] = (eval_weights[non_zero_eval_indices] - eval_weights[non_zero_eval_indices].min()) / (eval_weights[non_zero_eval_indices].max() - eval_weights[non_zero_eval_indices].min())
            hold_weights[non_zero_indices] = (hold_weights[non_zero_hold_indices] - hold_weights[non_zero_hold_indices].min()) / (hold_weights[non_zero_hold_indices].max() - hold_weights[non_zero_hold_indices].min())
            eval_weights = eval_weights/eval_weights.sum()
            hold_weights = hold_weights/hold_weights.sum()
            
            # Set weights as computed from above.
            subtensor.set_weights(
                wallet = wallet,
                netuid = metagraph.netuid,
                uids = metagraph.uids.tolist(),
                weights = weights.tolist(),
                wait_for_inclusion = False,
                wait_for_finalization = False,
            )
            print ('Scores:', scores )
            print ('Weights:', weights.tolist())
                        
        # Handle keyboard interrupts, stops training gracefully.
        except (KeyboardInterrupt, SystemExit):
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
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network uid.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--base_alpha', type=float, default = 0.001, help='Block based moving alpha for setting weights.')
    parser.add_argument('--uids_per_epoch', type=int, default = 3, help='Number of miners to eval on each window.')
    parser.add_argument('--blocks_per_uid', type=int, default = 10, help='Number of blocks we spend evaluating each miner.')
    parser.add_argument('--window_size', type=int, default=50, help='Size of eval window used to evaluate the miner')
    parser.add_argument('--window_speed', type=int, default=5, help='Speed that eval window moves forward across series.')
    parser.add_argument('--temperature', type=int, default=20, help='How steep the exponentiation is.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    bt.wallet.add_args( parser )
    bt.subtensor.add_args( parser )
    config = bt.config( parser )   
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/' # Fix this value.
    main( config ) 