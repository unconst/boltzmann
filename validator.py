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
                try: next_buckets.append(self.config.bucket if not self.config.remote else self.subtensor.get_commitment( self.config.netuid, uid ))
                except: next_buckets.append(None)    
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
                self.eval_window = self.current_window - 2
                
                # Download the slices for the window.
                logger.info(f"\tDownloading slices from previous window: { self.eval_window }")
                start_time = time.time()
                slice_files = await download_slices_for_buckets_and_windows(
                    buckets = self.buckets,
                    windows = [ self.eval_window ]
                )
                downloaded_per_step = sum([len(slice_files[k]) for k in slice_files])
                logger.info(f"\t\tDownloaded {downloaded_per_step} slices for previous window: { self.eval_window } in {time.time() - start_time} seconds")
                if downloaded_per_step == 0:
                    # Start on next window.
                    while self.current_window == self.step_window:
                        await asyncio.sleep(0.1)
                    continue
                
                # Get the indices for the current window that the miners will be evaluated on.
                indices = await get_indices_for_window(
                    model = self.model, 
                    seed = self.window_to_seed( self.eval_window + 1 ), # Seed index from current window
                    compression = self.hparams.compression
                )
                
                # Retrieve the list of slices for the evaluation window
                slices_for_eval_window = slice_files.get(self.eval_window, [])
                # Eval the downloaded slices.
                random.shuffle(slices_for_eval_window)
                # Evaluate miners until the time window expires.
                for idx, slice_info in enumerate(slices_for_eval_window):
                    try:
                        # Break the loop if the current time window has ended.
                        if self.step_window != self.current_window:
                            break

                        # ------------------------
                        # Step 1: Retrieve miner information and model update (slice).
                        # ------------------------
                        # Get the miner's UID (unique identifier) based on their hotkey.
                        uid = self.metagraph.hotkeys.index(slice_info.hotkey)
                        # Load the miner's model update (slice) from the temporary file.
                        # 'values' is a dictionary containing parameter updates for specific indices.
                        values = torch.load(
                            slice_info.temp_file,
                            map_location=torch.device(self.model.device),
                            weights_only=True
                        )
                        # Calculate the offset for the evaluation dataset based on the window parameters.
                        offset = self.eval_window * self.hparams.window_length * self.hparams.window_speed

                        # ------------------------
                        # Step 2: Load the evaluation dataset for this miner.
                        # ------------------------
                        # Record the start time for loading the evaluation dataset.
                        start_time = time.time()
                        logger.info(
                            f"\tLoading {self.hparams.validator_pages_per_eval} pages for miner uid: {uid} "
                            f"at window: {self.eval_window} and offset: {offset}"
                        )
                        # Generate a list of pages for the evaluation dataset.
                        # The pages are selected based on the miner's UID and the window, ensuring miner-specific evaluation data.
                        eval_pages = random.sample(
                            await AsyncSubsetFineWebEdu2Loader.next_pages(
                                offset=offset,
                                n_pages=self.hparams.validator_window_eval_size,
                                seed=uid
                            ),
                            self.hparams.validator_pages_per_eval
                        )
                        # Create the evaluation dataset using the selected pages.
                        eval_dataset = await AsyncSubsetFineWebEdu2Loader.create(
                            batch_size=self.config.actual_batch_size,
                            sequence_length=self.hparams.sequence_length,
                            pages_info=eval_pages,
                            tokenizer=self.hparams.tokenizer
                        )
                        logger.info(
                            f"\t\tLoaded eval dataset pages: {[p[1] for p in eval_pages]} "
                            f"in {time.time() - start_time} seconds"
                        )

                        # ------------------------
                        # Step 3: Compute the baseline loss on the evaluation dataset.
                        # ------------------------
                        logger.info(f"\tComputing baseline loss for miner uid: {uid}")
                        self.model.eval()  # Set the model to evaluation mode.
                        L_base = 0  # Initialize the baseline loss.
                        # Iterate over the evaluation dataset and compute the loss.
                        for batch in eval_dataset:
                            # Convert batch to tensor and move to the appropriate device.
                            input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                            # Clone input_ids to create labels; replace padding tokens with -100 to ignore them in loss computation.
                            labels = input_ids.clone()
                            labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                            # Compute the loss without gradient computation.
                            with torch.no_grad():
                                outputs = self.model(input_ids=input_ids, labels=labels)
                                L_base += outputs.loss.item()
                        # Compute the average baseline loss.
                        L_base /= len(eval_dataset)
                        logger.info(
                            f"\t\tFinished computing baseline loss in {time.time() - start_time} seconds"
                        )

                        # ------------------------
                        # Step 4: Apply the miner's update to the model.
                        # ------------------------
                        # Save the original parameter values at the specified indices to restore later.
                        previous_values = {}
                        # The 'indices' dictionary contains the indices of the parameters that the miner's update affects.
                        # It should be defined elsewhere in the code, mapping parameter names to index tensors.
                        # For example: indices = {'weight': tensor([0, 5, 10]), 'bias': tensor([1, 2])}
                        # Apply the miner's update by replacing parameter values at the specified indices.
                        for name, param in self.model.named_parameters():
                            if name in indices:
                                # Save the original values.
                                idx = indices[name].to(self.model.device)
                                previous_values[name] = param.data.view(-1)[idx].clone()
                                # Apply the miner's update.
                                param.data.view(-1)[idx] = values[name].to(self.model.device)
                        # Record the start time for computing the updated loss.
                        start_time = time.time()
                        logger.info(f"\tComputing updated loss after applying miner's update for uid: {uid}")
                        self.model.eval()  # Ensure the model is in evaluation mode.

                        # ------------------------
                        # Step 5: Compute the updated loss on the evaluation dataset.
                        # ------------------------
                        L_updated = 0  # Initialize the updated loss.
                        # Iterate over the evaluation dataset and compute the loss.
                        for batch in eval_dataset:
                            input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                            labels = input_ids.clone()
                            labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                            with torch.no_grad():
                                outputs = self.model(input_ids=input_ids, labels=labels)
                                L_updated += outputs.loss.item()
                        # Compute the average updated loss.
                        L_updated /= len(eval_dataset)
                        logger.info(
                            f"\t\tFinished computing updated loss in {time.time() - start_time} seconds"
                        )

                        # ------------------------
                        # Step 6: Revert the model to its original state.
                        # ------------------------
                        # Restore the original parameter values.
                        for name, param in self.model.named_parameters():
                            if name in previous_values:
                                idx = indices[name].to(self.model.device)
                                param.data.view(-1)[idx] = previous_values[name]

                        # ------------------------
                        # Step 7: Compute the normalized loss reduction as the reward.
                        # ------------------------
                        # Calculate the loss reduction.
                        delta_L = L_base - L_updated
                        # Compute the normalized loss reduction (reward).
                        r_i = delta_L / L_base  # Normalized reward.
                        # Ensure the reward is non-negative.
                        r_i = max(r_i, 0)

                        # ------------------------
                        # Step 8: Update the moving average rewards and weights.
                        # ------------------------
                        # Update the moving average reward for the miner.
                        self.rewards[uid] = (
                            self.hparams.validator_moving_alpha * r_i +
                            (1 - self.hparams.validator_moving_alpha) * self.rewards[uid]
                        )
                        # Recompute the weights using a softmax function with temperature.
                        # This determines the influence of each miner's update on the master model.
                        self.weights[self.rewards != 0] = torch.softmax(
                            self.rewards[self.rewards != 0] * self.hparams.validator_weights_temperature, dim=0
                        )
                        # Log the results if using Weights and Biases.
                        if self.config.use_wandb:
                            wandb.log({
                                f"R/{uid}": r_i,
                                f"R/mv_{uid}": self.rewards[uid].item(),
                                f"W/{uid}": self.weights[uid].item(),
                            })
                        # Log the rewards and weights.
                        logger.info(
                            f"\tReward for miner uid {uid}: {r_i}, "
                            f"moving average reward: {self.rewards[uid]}, "
                            f"weights: {self.weights[self.rewards != 0]}"
                        )

                        # ------------------------
                        # Step 9: Clean up and free memory.
                        # ------------------------
                        # Delete the values to free up memory.
                        del values
                        # Zero out the gradients to prevent accumulation.
                        self.model.zero_grad()

                    except Exception as e:
                        logger.error(f"Miner evaluation failed with error: {e}, setting reward to zero.")
                        # If an error occurs, set the miner's reward to zero using the moving average formula.
                        self.rewards[uid] = (
                            self.hparams.validator_moving_alpha * 0 +
                            (1 - self.hparams.validator_moving_alpha) * self.rewards[uid]
                        )
                        # Recompute the weights using the updated rewards.
                        self.weights[self.rewards != 0] = torch.softmax(
                            self.rewards[self.rewards != 0] * self.hparams.validator_weights_temperature, dim=0
                        )
                        # Optionally, log the zero reward.
                        if self.config.use_wandb:
                            wandb.log({
                                f"R/{uid}": 0,
                                f"R/mv_{uid}": self.rewards[uid].item(),
                                f"W/{uid}": self.weights[uid].item(),
                            })
                
                # Apply slices to the model from the previous window.
                logger.info(f"\tApplying slices from previous window: { self.eval_window } to model.")
                start_time = time.time()
                eval_slices = await apply_slices_to_model( 
                    model = self.model, 
                    window = self.eval_window , # Get files from previous window.
                    seed = self.window_seeds[ self.step_window ], # Use seed as the hash of the current window.
                    compression = self.hparams.compression
                )
                applied_per_step = len(eval_slices)
                logger.info(f"\t\tApplied {applied_per_step} from previous window: { self.eval_window } with seed: { self.window_seeds[ self.step_window ] } in {time.time() - start_time} seconds")
                
                # Delete lingering files 
                logger.info(f"\tCleaning space.")
                start_time = time.time()
                await delete_files_before_window( window_max = self.current_window - self.hparams.max_history )
                logger.info(f"\t\tFinished cleaning space in {time.time() - start_time} seconds.")
                
            
                # Ensure window is over.
                while self.current_window == self.step_window:
                    await asyncio.sleep(0.1)
                                        
                # Set temperatured weights on the chain.
                if self.current_window % 100 == 0: 
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
