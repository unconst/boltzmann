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
        parser.add_argument('--project', type=str, default='QZWXEC', help='Optional wandb project name')
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
        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model.to(self.config.device)
        self.model.eval()
        
        # Init buckets.
        self.buckets = []
        for uid in self.metagraph.uids:
            try: self.buckets.append('decis')
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
        self.rewards = torch.zeros( 256, dtype = torch.float32 ) 
        self.weights = torch.zeros( 256, dtype = torch.float32 ) 
        print ( self.hparams )
        
    async def update(self):
        while not self.stop_event.is_set():                          # Loop until stop_event is set
            self.subtensor = bt.subtensor(config=self.config)        # Reinitialize subtensor with current config
            nxt_meta = self.subtensor.metagraph(self.config.netuid)  # Get the new metagraph for the given netuid
            self.hparams = load_hparams()                            # Reload hyperparameters
            next_buckets = []                                        # Initialize the next_buckets list
            for uid in nxt_meta.uids:                                # Iterate over new metagraph uids
                try: next_buckets.append('decis')                    # Append 'decis' to next_buckets
                except: next_buckets.append(None)                    # Append None if an exception occurs
            self.buckets = next_buckets                              # Update self.buckets with next_buckets
            for idx, hotkey in enumerate(self.metagraph.hotkeys):    # Iterate over current metagraph hotkeys
                if hotkey != nxt_meta.hotkeys[idx]:                  # Check if hotkey has changed in the new metagraph
                    self.rewards[idx] = 0                            # Reset rewards for the changed hotkey
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
                # Start step.
                logger.info('\n' + '-' * 40 + f' Step: {self.global_step} ' + '-' * 40)
                logger.info(f"Step: {self.global_step}, Window: {self.current_window}, "
                            f"Block: {self.current_block}, Time: {int(time.time())}")
                self.global_step += 1
                self.step_window = self.current_window
                
                # Download files.    
                logger.info(f"\tDownloading slices from last window: {self.last_window}")
                start_time = time.time()
                slice_files = await download_slices_for_buckets_and_windows(
                    buckets = self.buckets,
                    windows = [self.last_window]
                )
                downloaded_per_step = sum([len(slice_files[k]) for k in slice_files])
                logger.info(f"\t\tDownloaded {downloaded_per_step} slices from last window: {self.last_window} in {time.time() - start_time} seconds")
                
                # Apply slices to the model from the previous window.
                logger.info(f"\tApplying slices from last window: {self.last_window} to model.")
                start_time = time.time()
                slice_files = await apply_slices_to_model( 
                    model = self.model, 
                    window = self.last_window, # Get files from previous window.
                    seed = self.window_seeds[ self.current_window ], # Use seed as the hash of the current window.
                    compression = self.hparams.compression
                )
                applied_per_step = len(slice_files)
                logger.info(f"\t\tApplied {applied_per_step} last window slices to model in {time.time() - start_time} seconds")
                
                # Delete lingering files 
                logger.info(f"\tCleaning space.")
                start_time = time.time()
                await delete_files_before_window( window_max = self.current_window - self.hparams.max_history )
                logger.info(f"\t\tFinished cleaning space in {time.time() - start_time} seconds.")
                
                # Eval until slice changes.
                random.shuffle(slice_files)
                for idx, slice_filename in enumerate(slice_files):
                    try:
                        # Break if we run out of time.
                        if self.last_window != self.current_window - 1: break
                        logger.info(f"\tEval Step: {idx}")

                        # Get random UID to eval.
                        miner_hotkey = slice_filename.split('-')[2].split('.')[0]
                        miner_uid = self.metagraph.hotkeys.index(miner_hotkey)
                        miner_values = torch.load(slice_filename, map_location=torch.device(self.model.device), weights_only = True)
                        miner_indices = await get_indices_for_window(
                            model = self.model, 
                            window = self.last_window, # Get files from previous window.
                            seed = self.window_seeds[ self.current_window ], # Seed index from current window
                            compression = self.hparams.compression
                        )
                        logger.info(f"\t\tUid: {miner_uid}")
                        logger.info(f"\t\tHotkey: {miner_hotkey}")
                        logger.info(f"\t\tWindow: {self.last_window}")
                        logger.info(f"\t\tFilename: {slice_filename}")
                        logger.info(f"\t\tLoaded values and indices.")
                        # Load pages from the miner eval window - (eval_pages_start, eval_pages_end)
                        #   eval_pages_start : ( window_idx * window_length * window_speed )
                        #   eval_pages_end   : ( window_idx * window_length * window_speed ) + window_eval_size
                        pages = random.sample(
                            await AsyncSubsetFineWebEdu2Loader.next_pages(
                                offset = self.last_window * self.hparams.window_length * self.hparams.window_speed,
                                n_pages = self.hparams.validator_window_eval_size,
                                seed = miner_uid
                            ), 
                            self.hparams.validator_pages_per_eval
                        )
                        dataset = await AsyncSubsetFineWebEdu2Loader.create(
                            batch_size = self.config.actual_batch_size,
                            sequence_length = self.hparams.sequence_length,
                            pages_info = pages,
                            tokenizer = self.hparams.tokenizer
                        )
                        logger.info(f"\t\tEval pages: {[p[1] for p in pages]} dataset for offset: {self.current_block * self.hparams.window_speed} ")
                        
                        # Zero grad on the model and compute loss.
                        self.model.zero_grad()
                        for idx, batch in enumerate(dataset):
                            input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                            labels = input_ids.clone()
                            labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                            with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):
                                outputs = self.model(input_ids=input_ids, labels=labels)
                                loss = outputs.loss
                                loss.backward()  # Compute gradients
                        logger.info(f"\t\tComputed gradient on model for pages")

                        # Collect the gradients
                        gradients = {}
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                gradients[name] = param.grad.detach().clone()
                            else:
                                # If the parameter did not receive a gradient, we set it to zero
                                gradients[name] = torch.zeros_like(param.data)

                        # Step 4: Flatten the gradients and the miner's update (slice values)
                        step_start_time = time.time()
                        gradient_vector = []
                        update_vector = []
                        for name in sorted(self.model.state_dict().keys()):
                            if name in gradients and name in miner_values:
                                # If there are gradients and values, add them.
                                grad = gradients[name].view(-1)
                                update = torch.zeros_like(grad)
                                indices = miner_indices[name].to(self.model.device)
                                values = miner_values[name].to(self.model.device)
                                update[indices] = values
                                gradient_vector.append(grad)
                                update_vector.append(update)
                            else:
                                # If no gradient or update, append zeros
                                size = self.model.state_dict()[name].numel()
                                gradient_vector.append(torch.zeros(size, device=self.model.device))
                                update_vector.append(torch.zeros(size, device=self.model.device))
                        # Concatenate all parameter gradients and updates into single tensors
                        gradient_vector = torch.cat(gradient_vector)
                        update_vector = torch.cat(update_vector)
                        step_end_time = time.time()

                        # Compute reward.
                        dot_product = torch.dot(gradient_vector, update_vector)
                        update_norm = torch.norm(update_vector)
                        regularization = self.hparams.validator_norm_regularization * update_norm.item()
                        reward = -dot_product.item() - regularization
                        # Update rewards vector with moving average
                        self.rewards[miner_uid] = (reward * self.hparams.validator_rewards_alpha) + ((1 - self.hparams.validator_rewards_alpha) * self.rewards[miner_uid])
                        # Recompute weights from rewards.
                        self.weights[ self.rewards != 0 ] = torch.softmax( self.rewards[ self.rewards != 0 ] * self.hparams.validator_weights_temperature, dim=0)
                        if self.config.use_wandb:
                            wandb.log({
                                f"R/{miner_uid}": self.rewards[miner_uid],
                                f"R/moving_{miner_uid}": reward,
                                f"W/{miner_uid}": self.weights[miner_uid],
                                f"update_norm/{miner_uid}": update_norm.item(),
                                f"regularization/{miner_uid}": regularization,
                                f"dot_product/{miner_uid}": dot_product.item(),
                                f"reward/{miner_uid}": reward,
                            })
                        logger.info(f'\t\tupdate_norm: {update_norm}, regularization: {regularization}, dot_product: {dot_product}, reward: {reward}, weights: {self.weights[miner_uid]}')
                                            
                        # Clean up to free memory
                        del gradients
                        del gradient_vector
                        del update_vector
                        del miner_indices
                        del miner_values
                        
                    # We can't download the slice for the miner.    
                    except Exception as e:
                        logger.error(f"Miner eval failed with error: {e}, setting score of zero.")
                        # Update rewards vector with moving average
                        self.rewards[miner_uid] = (reward * self.hparams.validator_rewards_alpha) + ((1 - self.hparams.validator_rewards_alpha) * self.rewards[miner_uid])
                        # Recompute weights from rewards.
                        self.weights[ self.rewards != 0 ] = torch.softmax( self.rewards[ self.rewards != 0 ] * self.hparams.validator_weights_temperature, dim=0)
                    
                # Wait until we are on a new window (if need be)
                while self.current_window == self.step_window:
                    await asyncio.sleep(0.1)
                self.last_window = self.current_window - 1
                                        
                # Set temperatured weights on the chain.
                if self.current_window % 10 == 0: 
                    self.subtensor.set_weights(
                        wallet = self.wallet,
                        netuid = self.config.netuid,
                        uids = self.metagraph.uids,
                        weights = self.weights[ self.metagraph.uids ],
                        wait_for_inclusion = False,
                        wait_for_finalization = False,
                    )
                        
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
                logger.info(f"\t\tNew window: {self.current_window}")
        # Subscribe to block headers with the custom handler
        bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)
            
if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.run())
