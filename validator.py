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
# fmt: off

# Global imports.
import os
import sys 
import time
import wandb
import torch
import random
import asyncio
import argparse
import threading
import traceback
from tqdm import tqdm
import bittensor as bt
from typing import List
import torch.optim as optim
from dotenv import dotenv_values
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import local files.
from common import *
from hparams import load_hparams
from dataset import AsyncSubsetFineWebEdu2Loader

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Validator:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Validator script')
        parser.add_argument('--project', type=str, default='aesop2', help='Optional wandb project name')
        parser.add_argument('--netuid', type=int, default=220, help='Bittensor network UID.')
        parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
        parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        config = bt.config(parser)
        config.subtensor.network = 'test'
        config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
        if config.debug: debug()
        if config.trace: trace()
        return config

    def __init__(self):
        # Init config.
        self.config = Validator.config()
        logger.info('\n' + '-' * 40 + ' Config ' + '-' * 40)
        logger.info(self.config)

        # Init bittensor objects.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise ValueError(f'Wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}')
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info('\n' + '-' * 40 + ' Objects ' + '-' * 40)
        logger.info(f'\nWallet: {self.wallet}\nSubtensor: {self.subtensor}\nMetagraph: {self.metagraph}\nUID: {self.uid}')

        # Init bucket.
        try:
            if self.config.bucket != self.subtensor.get_commitment(self.config.netuid, self.uid):
                raise ValueError('')
        except:
            self.subtensor.commit(self.wallet, self.config.netuid, self.config.bucket)
        logger.info('Bucket:' + self.config.bucket)

        # Init Wandb.
        if self.config.use_wandb:
            # Delete all runs with my name and create a new one.
            try:
                [run.delete() for run in wandb.Api().runs(path=self.config.project)
                 if run.name == f'V{self.uid}' and logger.info(f'Deleting old run: {run}')]
            except: pass
            wandb.init(project=self.config.project, resume='allow', name=f'V{self.uid}', config=self.config)

        # Init model.
        logger.info('\n' + '-' * 40 + ' Hparams ' + '-' * 40)
        self.hparams = load_hparams()
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        #self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model = LlamaForCausalLM.from_pretrained('TinyLlama/TinyLlama_v1.1')
        self.model.to(self.config.device)
        self.model.eval()
        
        # Init buckets.
        self.buckets = []
        for uid in self.metagraph.uids:
            # Use --remote to connect to other miners, other wise, only see's config.bucket.
            try: self.buckets.append(self.config.bucket if not self.config.remote else self.subtensor.get_commitment( self.config.netuid, uid ) )
            except: self.buckets.append(None)

        # Init run state.
        self.global_step = 0
        self.last_window = 0
        self.optimal_pages_per_step = 4
        self.current_block = self.subtensor.block
        self.current_window = self.block_to_window( self.current_block )
        self.window_seeds = {self.current_window: self.window_to_seed( self.current_window) }
        self.block_event = asyncio.Event()
        self.new_window_event = asyncio.Event()
        self.stop_event = asyncio.Event()     
        self.step_scores = torch.zeros( 256, dtype = torch.float32 ) 
        self.scores = torch.zeros( 256, dtype = torch.float32 ) 
        self.weights = torch.zeros( 256, dtype = torch.float32 ) 
        self.sample_rate = 1.0
        print ( self.hparams )
        
    async def update(self):
        while not self.stop_event.is_set():                          # Loop until stop_event is set
            self.subtensor = bt.subtensor(config=self.config)        # Reinitialize subtensor with current config
            nxt_meta = self.subtensor.metagraph(self.config.netuid)  # Get the new metagraph for the given netuid
            self.hparams = load_hparams()                            # Reload hyperparameters
            next_buckets = []                                        # Initialize the next_buckets list
            for uid in nxt_meta.uids:                                # Iterate over new metagraph uids
                try: next_buckets.append(self.config.bucket if not self.config.remote else self.subtensor.get_commitment( self.config.netuid, uid ))
                except: next_buckets.append(None)    
            self.buckets = next_buckets                              # Update self.buckets with next_buckets
            for idx, hotkey in enumerate(self.metagraph.hotkeys):    # Iterate over current metagraph hotkeys
                if hotkey != nxt_meta.hotkeys[idx]:                  # Check if hotkey has changed in the new metagraph
                    self.scores[idx] = 0                             # Reset rewards for the changed hotkey
                    self.weights[idx] = 0                            # Reset weights for the changed hotkey
            self.metagraph = nxt_meta                                # Update self.metagraph with new_metagraph
            await asyncio.sleep(60)                                  # Sleep for 60 seconds before the next iteration

    async def run(self):
        # Main loop.
        self.loop = asyncio.get_running_loop()
        self.update_task = asyncio.create_task(self.update())
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()

        while True:
            try:
                # Get the window we are evalling.
                eval_window = self.current_window - 1
                
                # Download the state for the eval window.
                await download_slices_for_buckets_and_windows(
                    buckets = self.buckets,
                    windows = [ eval_window ],
                    key = 'state'
                )
                logger.info(f"{eval_window}: Downloaded the state for the eval window: {eval_window} with current: {self.current_window}")
                
                # Download the delta for the eval window.
                eval_slices = await download_slices_for_buckets_and_windows(
                    buckets = self.buckets,
                    windows = [ eval_window ],
                    key = 'delta'
                ) 
                logger.info(f"{eval_window}: Downloaded the delta from the eval window: {eval_window} with current: {self.current_window}")
                
                # Check for slices.
                if len(list(eval_slices.keys())) == 0:
                    logger.info(f"{eval_window}: No slices to eval, continue ...")
                    while self.current_window == eval_window: 
                        await asyncio.sleep(0.1)
                    continue
                logger.info(f"{eval_window}: Found {len(eval_slices[ eval_window ])} deltas for {eval_window}")
                
                # Applied the model state state for the eval window.
                await apply_slices_to_model( 
                    model = self.model, 
                    window = eval_window,
                    seed = eval_window,
                    compression = self.hparams.compression,
                    key = 'state',
                )
                logger.info(f"{eval_window}: Applied the state for the eval window: {eval_window}")
                
                # Attain the indicies for the eval window.
                indices = await get_indices_for_window(
                    model = self.model,
                    seed = eval_window,
                    compression = self.hparams.compression
                ) 
                logger.info(f"{eval_window}: Attained the indices for the eval window: {eval_window}")
                               
                # Select the random slice to eval.
                eval_slice_info = random.choice( eval_slices[ eval_window ] )                
                logger.info(f"{eval_window}: Selected {eval_slice_info} to eval this window.")

                # Attain the UID of this slice.                
                try: eval_uid = self.metagraph.hotkeys.index(eval_slice_info.hotkey)
                except ValueError:
                    logger.warning(f"Hotkey {eval_slice_info.hotkey} not found in metagraph hotkeys.")
                    continue                                
                logger.info(f"{eval_window}: Evaluating UID: {eval_uid}")
                
                # Load the slice data for this UID
                eval_slice_data = await get_slices( eval_slice_info.temp_file, self.model.device )                
                logger.info(f"{eval_window}: Loaded the slice data.")

                # Download the eval page for this uid.,
                eval_page = await AsyncSubsetFineWebEdu2Loader.next_pages(
                    offset = eval_window,
                    n_pages = 1,
                    seed = eval_uid
                )                
                eval_dataset = await AsyncSubsetFineWebEdu2Loader.create(
                    batch_size = self.config.actual_batch_size,
                    sequence_length = self.hparams.sequence_length,
                    pages_info = eval_page,
                    tokenizer = self.hparams.tokenizer
                )                
                logger.info(f"{eval_window}: Downloaded the eval page: {eval_page[0][1]} for this window.")
  
                # Accumulate gradients from this page.
                self.model.zero_grad()
                self.model.eval()
                total_loss = 0
                with torch.enable_grad():
                    for idx, batch in enumerate(eval_dataset):
                        input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                        with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):  # Enable autocasting
                            outputs = self.model(input_ids=input_ids, labels=labels)
                        total_loss += outputs.loss.item()
                        outputs.loss.backward()
                logger.info(f"{eval_window}: Accumulated gradients with loss: {total_loss} over over: {idx} steps.")

                # Compute the score for this slice.
                score = 0.0 
                for i, (name_i, param_i) in enumerate( self.model.named_parameters() ):
                    if param_i.grad is None: continue  # Skip parameters without gradients
                    idxs_i = indices[name_i].to(self.model.device)
                    grad_i = param_i.grad.view(-1).clone()[idxs_i].to(self.model.device) 
                    slice_i = eval_slice_data[name_i].view(-1).to(self.model.device)  # Shape: [num_params_in_subset]
                    theta_i = param_i.data.view(-1)[idxs_i]  # Shape: [num_params_in_subset]
                    delta_i = theta_i - slice_i
                    sim_i = torch.nn.functional.cosine_similarity(delta_i, grad_i, dim=0).item()
                    weight_i = param_i.data.view(-1)[idxs_i].norm().item() + 1e-8
                    score += weight_i * sim_i
                logger.info(f"{eval_window}: Computed score for this slice with score: {score}")           

                # Assign scores.
                self.step_scores[ eval_uid ] = score
                self.scores[ eval_uid ] = (1 - self.hparams.validator_moving_alpha) * score + self.hparams.validator_moving_alpha * self.scores[eval_uid]
                self.scores[ torch.isnan(self.scores) ] = 0
                valid_score_indices = torch.nonzero((self.scores != 0) & (~torch.isnan(self.scores))).squeeze().view(-1, 1)
                valid_scores = self.scores[valid_score_indices].view(-1, 1) if valid_score_indices.dim() == 1 else self.scores[valid_score_indices]
                if valid_scores.numel() > 0:
                    min_score = torch.min(valid_scores)
                    max_score = torch.max(valid_scores)
                    self.weights[valid_score_indices] = (valid_scores - min_score) / (max_score - min_score + 1e-8)
                logger.info(f"{eval_window}: Assign the scores and compute the weights.")  

                # Log and print scores.
                if self.config.use_wandb:
                    for uid_i in valid_score_indices:
                        wandb.log({
                            f"step_scores/{uid_i.item()}": self.step_scores[ uid_i ].item(),
                            f"moving_scores/{uid_i.item()}": self.scores[ uid_i ].item(),
                            f"weights/{uid_i.item()}": self.weights[ uid_i ].item(),
                            'self.sample_rate': self.sample_rate,
                        })
                for uid_i in valid_score_indices:
                    moving_score = self.scores[ uid_i ].item()
                    weight = self.weights[ uid_i ].item()
                    step_score = self.step_scores[ uid_i ].item()
                    logger.info(f"\t\tuid: {uid_i.item()}, step_scores: {step_score:.6f}, moving_score: {moving_score:.6f}, weight: {weight:.6f}")
                
                # Apply all deltas to the model state.
                await apply_slices_to_model( 
                    model = self.model, 
                    window = self.current_window - 1,
                    seed = self.current_window - 1,
                    compression = self.hparams.compression,
                    key = 'delta',
                )
                logger.info(f"{eval_window}: Applied deltas to the model for window: {eval_window}")

                # Finish step.
                while self.current_window == eval_window: 
                    await asyncio.sleep(0.1)
                continue
                logger.info(f"{eval_window}: Finished step.")                

                                                                
            # Catch keyboard interrrupt.
            except KeyboardInterrupt:
                logger.info("Training interrupted by user. Stopping the run.")
                self.stop_event.set()
                await self.update_task
                sys.exit(0)
            
            # Catch unknown.
            except Exception as e:
                logger.exception(f"Exception during training loop: {e}")
                continue

    # Returns the slice window based on a block.
    def block_to_window(self, block: int) -> int:
        return int(block / self.hparams.window_length)
    
    # Returns the slice window based on a block.
    def window_to_seed(self, window: int) -> int:
        return str( self.subtensor.get_block_hash( window * self.hparams.window_length ) )

    # A listener thread which posts the block event
    # when the chain announces a new block.
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            loop.call_soon_threadsafe(self.block_event.set)
            if self.block_to_window(self.current_block) != self.current_window:
                self.window_seeds[ self.block_to_window(self.current_block) ] = self.window_to_seed( self.block_to_window(self.current_block) )
                self.current_window = self.block_to_window(self.current_block)
                loop.call_soon_threadsafe(self.new_window_event.set)
                logger.info(f"-- New window: {self.current_window} -- ")
        # Run listener with retry.
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler); break
            except Exception as e:
                 # Wait for 5 seconds before retrying
                logger.error(f"Failed to subscribe to block headers: {e}.\nRetrying in 1 seconds...")
                time.sleep(1) 
            
if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.run())
