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
        parser.add_argument('--project', type=str, default='aesop2', help='Optional wandb project name')
        parser.add_argument('--netuid', type=int, default=220, help='Bittensor network UID.')
        parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
        parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--remote', action='store_true', help='Connect to other buckets')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--random', action='store_true', help='Train on random')
        parser.add_argument('--sync_state', action='store_true', help='Syncs the model state by pulling from the history.')
        parser.add_argument('--baseline', action='store_true', help='Dont perform syncing with other peers, just train.')
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
                for run in wandb.Api().runs(path=self.config.project):
                    if run.name == f'M{self.uid}':
                        logger.info(f'Deleting old run: {run}'); run.delete()
            except: pass
            wandb.init(project=self.config.project, resume='allow', name=f'M{self.uid}', config=self.config)

        # Init model.
        logger.info('\n' + '-' * 40 + ' Hparams ' + '-' * 40)
        self.hparams = load_hparams()
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        # self.model = LlamaForCausalLM.from_pretrained('TinyLlama/TinyLlama_v1.1')
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
            st = T()
            self.subtensor = bt.subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            self.hparams = load_hparams()
            next_buckets = []
            for uid in self.metagraph.uids:
                try: next_buckets.append(self.config.bucket if not self.config.remote else self.subtensor.get_commitment( self.config.netuid, uid ))
                except: next_buckets.append(None)    
            self.buckets = next_buckets    
            logger.info(f"{P(self.current_window, T() - st)} Updated global state.")
            await asyncio.sleep(60)

    async def run(self):
        # Main loop.
        self.loop = asyncio.get_running_loop()
        self.update_task = asyncio.create_task(self.update())
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()
        
        # Optionally sync the model state by pulling model states from the history.
        if self.config.sync_state:
            history_windows = [ self.current_window - i for i in range (self.hparams.max_history) ]
            state_slices = await download_slices_for_buckets_and_windows(
                buckets = self.buckets,
                windows = history_windows,
                key = 'state'
            )
            for window in tqdm(history_windows, desc="Syncing state"):
                await apply_slices_to_model( 
                    model = self.model, 
                    window = window,
                    seed = window,
                    compression = self.hparams.compression,
                    key = 'state'
                )
            torch.cuda.empty_cache()
            
        # Main training loop.
        while True:
            try:      
                # Start the window step.     
                logger.info('[bold]' + '\n' + '-' * 40 + f' Step: {self.global_step} ' + '-' * 40)
                self.global_step += 1
                start_step = T()
                window = self.current_window
                
                # Run for non-baseline miners.
                if not self.config.baseline:
                    st = T()
                    state_slices = await download_slices_for_buckets_and_windows(
                        buckets = self.buckets,
                        windows = [ window ],
                        key = 'state'
                    )
                    n_slices = len(state_slices[ window ]) if window in state_slices else 0
                    logger.info(f"{P(window, T() - st)}: Downloaded {n_slices} window states.")
                    
                    # Download the delta from the previous window.
                    st = T()
                    delta_slices = await download_slices_for_buckets_and_windows(
                        buckets = self.buckets,
                        windows = [ window - 1 ],
                        key = 'delta'
                    )       
                    n_slices = len(delta_slices[ window - 1  ]) if window - 1 in delta_slices else 0
                    logger.info(f"{P(window, T() - st)}: Download {n_slices} window deltas.")
                    
                    # Apply the state for the current window.
                    st = T()
                    await apply_slices_to_model( 
                        model = self.model, 
                        window = window,
                        seed = window,
                        compression = self.hparams.compression,
                        key = 'state'
                    )
                    logger.info(f"{P(window, T() - st)}: Applied window state.")
        
                # Download the page for the current window.
                st = T()
                pages = await DatasetLoader.next_pages(
                    offset = window,
                    n_pages = self.hparams.validator_window_eval_size,
                    seed = self.uid if not self.config.random else random.randint(0, 1000)
                )
                random.shuffle( pages )
                dataset = await DatasetLoader.create(
                    batch_size = self.config.actual_batch_size,
                    sequence_length = self.hparams.sequence_length,
                    pages_info = pages,
                    tokenizer = self.hparams.tokenizer
                )
                logger.info(f"{P(window, T() - st)}: Downloaded training page: [light_steel_blue]{[p[1] for p in pages]}[/light_steel_blue] random = {self.config.random}")

                # Accumualte gradients on the model applied to the base state.
                train_start = T()
                self.model.zero_grad(); self.model.eval()
                total_loss = 0.0
                full_steps = 0; total_steps = 0; 
                exhuasted_window = False
                for batch in dataset:
                    total_steps += 1
                    if random.random() < self.sample_rate and not exhuasted_window:
                        full_steps += 1
                        input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                        with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):  # Enable autocasting
                            outputs = self.model(input_ids=input_ids, labels=labels)
                        total_loss += outputs.loss.item()
                        outputs.loss.backward()     
                        if window != self.current_window and not self.config.baseline: exhuasted_window = True; continue
                if self.hparams.grad_clip: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                step_loss = total_loss/(full_steps+1)
                train_duration = T() - train_start
                tokens_per_step = self.hparams.sequence_length * self.config.actual_batch_size * (full_steps + 1)
                tokens_per_second =  tokens_per_step / train_duration
                logger.info(f"{P(window, train_duration)} Accumulated gradients:")
                logger.info(f"{P(window, train_duration)} \tTotal steps: [tan]{full_steps}/{total_steps}[/tan], Rate: [tan]{(full_steps/total_steps):.2f}[/tan], Target: [tan]{self.sample_rate:.2f}[/tan]")
                logger.info(f"{P(window, train_duration)} \tTotal tokens: [tan]{tokens_per_step}[/tan], Tokens per second: [tan]{tokens_per_second:.2f}[/tan]")
                logger.info(f"{P(window, train_duration)} \tLoss: [tan]{step_loss}[tan]")
                if exhuasted_window: self.sample_rate = max(0.0001, self.sample_rate * 0.95)
                else: self.sample_rate = min(1, self.sample_rate * 1.05)

                # Run for non-baseline nodes.
                if not self.config.baseline:
                    # Upload the delta for the previous window.
                    st = T()
                    await upload_slice_for_window(
                        bucket = self.config.bucket, 
                        model = self.model, 
                        window = window,
                        seed = window,
                        wallet = self.wallet, 
                        compression = self.hparams.compression,
                        key = 'delta'
                    )                
                    logger.info(f"{P(window, T() - st)}: Uploaded the delta.")
                    
                    # Apply the delta from the previous window.
                    st = T()
                    await apply_slices_to_model(
                        model = self.model, 
                        window = window - 1,
                        seed = window - 1,
                        compression = self.hparams.compression,
                        key = 'delta'
                    )         
                    logger.info(f"{P(window, T() - st)}: Applied window delta.")
                
                    # Upload the state for the current window.
                    st = T()
                    await upload_slice_for_window(
                        bucket = self.config.bucket, 
                        model = self.model, 
                        window = window + 1,
                        seed = window + 1, 
                        wallet = self.wallet, 
                        compression = self.hparams.compression,
                        key = 'state',
                    )
                    logger.info(f"{P(window, T() - st)}: Uploaded the state.")
                    
                    # Clean file history.
                    st = T()
                    await delete_files_before_window( window_max = window - self.hparams.max_history, key = 'state')
                    await delete_files_before_window( window_max = window - self.hparams.max_history, key = 'delta')
                    await delete_files_from_bucket_before_window( bucket = self.config.bucket, window_max = window - self.hparams.max_history, key = 'state' )
                    await delete_files_from_bucket_before_window( bucket = self.config.bucket, window_max = window - self.hparams.max_history, key = 'delta' )
                    logger.info(f"{P(window, T() - st)}: Cleaned file history.")
                    
                    # Wait until we are on a new window.
                    end_step = T()
                    while self.current_window == window:
                        await asyncio.sleep(0.1)
                    window_time_delta = self.window_time - end_step
                    window_delta_str = f"[red]{window_time_delta:.2f}[/red]" if window_time_delta < 0 else f"[green]+{window_time_delta:.2f}[/green]"
                    logger.info(f"{P(window, end_step - start_step)}[{window_delta_str}]: Finished step.")
                    if self.config.use_wandb:
                        wandb.log({
                            f"loss": step_loss,
                            f"tokens_per_step": tokens_per_step,
                            f"tokens_per_second": tokens_per_second,
                            f"sample_rate": self.sample_rate,
                            f"utilization": train_duration / (end_step - start_step),
                            f"learning_rate": self.scheduler.get_last_lr()[0]
                        })
                                
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
                self.window_duration = T() - self.window_time if hasattr(self, 'window_time') else 0
                self.window_time = T()
                loop.call_soon_threadsafe(self.new_window_event.set)
                logger.info(f"{P(self.current_window, self.window_duration)} New Window.")
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
