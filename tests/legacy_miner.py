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
import math
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
from dataset import DatasetLoader

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Miner:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--project', type=str, default='QZWXEC', help='Optional wandb project name')
        parser.add_argument('--netuid', type=int, default=220, help='Bittensor network UID.')
        parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
        parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--remote', action='store_true', help='Connect to other buckets')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--random', action='store_true', help='Train on random')
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
        self.config = Miner.config()
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
                 if run.name == f'M{self.uid}-{"r" if self.config.random else ""}' and logger.info(f'Deleting old run: {run}')]
            except: pass
            wandb.init(project=self.config.project, resume='allow', name=f'M{self.uid}', config=self.config)

        # Init model.
        logger.info('\n' + '-' * 40 + ' Hparams ' + '-' * 40)
        self.hparams = load_hparams()
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        # self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model = LlamaForCausalLM.from_pretrained('TinyLlama/TinyLlama_v1.1')
        self.model.to(self.config.device)
        self.model.train()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,  # Peak learning rate
            betas=(self.hparams.optimizer_beta1, self.hparams.optimizer_beta2),  # B1 and B2
            weight_decay=self.hparams.optimizer_weight_decay,  # Weight decay
            foreach=True,  # more memory usage, but faster
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.hparams.cosine_epoch_length,
            eta_min=self.hparams.eta_min, last_epoch=-1
        )

        # Init buckets.
        self.buckets = []
        for uid in self.metagraph.uids:
            # Use --remote to connect to other miners, other wise, only see's config.bucket.
            try: self.buckets.append(self.config.bucket if not self.config.remote else self.subtensor.get_commitment( self.config.netuid, uid ) )
            except: self.buckets.append(None)

        # Init run state.
        self.global_step = 0
        self.sample_rate = 1.0
        self.current_block = self.subtensor.block
        self.current_window = self.block_to_window( self.current_block )
        self.window_seeds = {self.current_window: self.window_to_seed( self.current_window) }
        self.new_block_event = asyncio.Event()
        self.new_window_event = asyncio.Event()
        self.stop_event = asyncio.Event()    
        self.last_full_steps = self.hparams.desired_batch_size // self.config.actual_batch_size    
        print ( self.hparams )
        
    async def update(self):
        while not self.stop_event.is_set():
            logger.info(f"\tUpdating global state.")
            start_time = time.time()
            self.subtensor = bt.subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            self.hparams = load_hparams()
            next_buckets = []
            for uid in self.metagraph.uids:
                try: next_buckets.append(self.config.bucket if not self.config.remote else self.subtensor.get_commitment( self.config.netuid, uid ))
                except: next_buckets.append(None)    
            self.buckets = next_buckets    
            logger.info(f"\t\tUpdated global state in {time.time() - start_time} seconds.")
            await asyncio.sleep(60)

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
                global_step_start_time = time.time()
                self.step_window = self.current_window
                self.global_step += 1

                # Download files.    
                logger.info(f"\tDownloading slices from previous window: {self.step_window - 1}")
                start_time = time.time()
                slice_files = await download_slices_for_buckets_and_windows(
                    buckets = self.buckets,
                    windows = [self.step_window - 1]
                )
                downloaded_per_step = sum([len(slice_files[k]) for k in slice_files])
                logger.info(f"\t\tDownloaded {downloaded_per_step} slices for previous window: {self.step_window - 1} in {time.time() - start_time} seconds")
                
                # Apply slices to the model from the previous window.
                logger.info(f"\tApplying slices from previous window: {self.step_window - 1} to model.")
                start_time = time.time()
                slice_files = await apply_slices_to_model( 
                    model = self.model, 
                    window = self.step_window - 1, # Get files from previous window.
                    seed = self.window_seeds[ self.step_window ], # Use seed as the hash of the current window.
                    compression = self.hparams.compression
                )
                applied_per_step = len(slice_files)
                logger.info(f"\t\tApplied {applied_per_step} from previous window: {self.step_window - 1} with seed: { self.window_seeds[ self.step_window ] } in {time.time() - start_time} seconds")
                
                # Train for performance on the current window.
                # Load pages from the current eval window. The validators will sample pages from (eval_pages_start, eval_pages_end)
                #   eval_pages_start : ( window_idx * window_length * window_speed )
                #   eval_pages_end   : ( window_idx * window_length * window_speed ) + window_eval_size
                start_time = time.time()
                offset = self.step_window * self.hparams.window_length * self.hparams.window_speed
                seed = self.uid if not self.config.random else random.randint(0, 1000)
                logger.info(f"\tLoading {self.hparams.validator_window_eval_size} pages for current window: { self.step_window } and offset: {offset} and uid: {self.uid} and seed: {seed}")
                pages = await DatasetLoader.next_pages(
                    offset = offset,
                    n_pages = self.hparams.validator_window_eval_size,
                    seed = seed
                )
                random.shuffle( pages ) 
                dataset = await DatasetLoader.create(
                    batch_size = self.config.actual_batch_size,
                    sequence_length = self.hparams.sequence_length,
                    pages_info = pages,
                    tokenizer = self.hparams.tokenizer
                )
                pages_per_step = len(pages)
                logger.info(f"\t\tLoaded dataset pages: {[p[1] for p in pages]} in {time.time() - start_time} seconds")

                # Train the model on the current page.
                logger.info(f"\tTraining on pages: {[p[1] for p in pages]} with sample rate: {self.sample_rate}")
                start_time = time.time()
                torch.cuda.empty_cache()  # Empty cache going into the training step.
                self.optimizer.zero_grad()  # Clear any lingering grads.
                total_loss = 0.0
                exhuasted_window = False
                self.full_steps = 0
                for idx, batch in enumerate(dataset):
                    # Randomly sample every sample_rate examples
                    if random.random() < self.sample_rate:
                        self.full_steps += 1
                        input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                        with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):  # Enable autocasting
                            outputs = self.model(input_ids=input_ids, labels=labels)
                        total_loss += outputs.loss.item()
                        loss = outputs.loss / (self.last_full_steps + 1)  # Divide by number of accumulations.
                        loss.backward()
                        if self.step_window != self.current_window:
                            exhuasted_window = True
                            break

                # Apply step and clean memory.
                if self.hparams.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.grad_clip)
                self.optimizer.step()
                self.scheduler.step()  # Update the learning rate.
                self.optimizer.zero_grad()
                del input_ids, labels, outputs
                torch.cuda.empty_cache()

                # Calculate, print and log average loss
                self.last_full_steps = self.full_steps
                average_loss = total_loss / (self.full_steps + 1)
                total_time = time.time() - start_time
                tokens_per_step = self.hparams.sequence_length * self.config.actual_batch_size * (self.full_steps + 1)
                tokens_per_second =  tokens_per_step / total_time
                logger.info(f"\t\tTotal steps: {idx}, Applied: {self.full_steps}, Rate: {self.full_steps/(idx + 1)}, Sample Probability: {self.sample_rate}")
                logger.info(f"\t\tLoss: {average_loss}, learning_rate: {self.scheduler.get_last_lr()[0]}")
                logger.info(f"\t\tTraining completed in {total_time} seconds, Tokens per step: {tokens_per_step}, Tokens per second: {tokens_per_second}")
                if exhuasted_window:
                    self.sample_rate = max(0.0001, self.sample_rate * 0.95)
                else:
                    self.sample_rate = min(1, self.sample_rate * 1.05)
                
                # Wait until we are on a new window.
                while self.current_window == self.step_window:
                    await asyncio.sleep(0.1)

                # Upload our model slice to S3.
                logger.info(f"\tUploading for window: { self.step_window }")
                start_time = time.time()
                await upload_slice_for_window(
                    bucket = self.config.bucket, 
                    model = self.model, 
                    window = self.step_window, # Upload for the previous window 
                    seed = self.window_seeds[ self.step_window + 1 ], # Seed the index by the hash of the new window.
                    wallet = self.wallet, 
                    compression = self.hparams.compression
                )
                logger.info(f"\t\tFinished upload for window: {self.step_window} with seed: {self.window_seeds[ self.step_window + 1 ]} in {time.time() - start_time} seconds.")
                
                # Delete lingering files 
                logger.info(f"\tCleaning space.")
                start_time = time.time()
                await delete_files_before_window( window_max = self.current_window - self.hparams.max_history )
                await delete_files_from_bucket_before_window( bucket = self.config.bucket, window_max = self.current_window - self.hparams.max_history )
                logger.info(f"\t\tFinished cleaning space in {time.time() - start_time} seconds.")

                # Calculate and log global steps per second
                seconds_per_step = time.time() - global_step_start_time
                steps_per_second = 1 / seconds_per_step
                if self.config.use_wandb:
                    wandb.log({
                        "step_loss": average_loss,
                        "tokens_per_step": tokens_per_step,
                        "tokens_per_second": tokens_per_second,
                        "applied_per_step": applied_per_step,
                        "pages_per_step": pages_per_step,
                        "downloaded_per_step": downloaded_per_step,
                        "incentive": float(self.metagraph.I[self.uid]),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "seconds_per_step": seconds_per_step,
                        "steps_per_second": steps_per_second,
                        "sample_rate": self.sample_rate,
                    })
                    
                logger.info(f'\nGlobal step completed in {seconds_per_step} seconds\n')
                
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
        return int( block / self.hparams.window_length ) # floor
    
    # Returns the slice window based on a block.
    def window_to_seed(self, window: int) -> int:
        return str( self.subtensor.get_block_hash( window * self.hparams.window_length ) )

    # A listener thread which posts the block event
    # when the chain announces a new block.
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            loop.call_soon_threadsafe(self.new_block_event.set)
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
    asyncio.run(Miner().run())
