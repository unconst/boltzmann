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
import sys  # Required for logger output
import time
import wandb
import torch
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
from dataset import SubsetFineWebEdu2Loader

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Miner:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--project', type=str, default='220A', help='Optional wandb project name')
        parser.add_argument('--netuid', type=int, default=220, help='Bittensor network UID.')
        parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
        parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        config = bt.config(parser)
        config.subtensor.network = 'test'
        config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
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
        logger.info('\n' + '-' * 40 + 'Objects' + '-' * 40)
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
                 if run.name == f'M{self.uid}' and logger.info(f'Deleting old run: {run}')]
            except:
                pass
            wandb.init(project=self.config.project, resume='allow', name=f'M{self.uid}', config=self.config)

        # Init model.
        logger.info('\n' + '-' * 40 + 'Hparams' + '-' * 40)
        self.hparams = load_hparams()
        self.model = LlamaForCausalLM(config=self.hparams.model_config)
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
        for uid in tqdm(self.metagraph.uids):
            try:
                self.buckets.append('decis')
            except:
                self.buckets.append(None)

        # Init run state.
        self.global_step = 0
        self.optimal_pages_per_step = 4
        self.current_block = self.subtensor.block
        self.current_window = self.block_to_window( self.current_block )
        self.block_event = asyncio.Event()
        self.new_window_event = asyncio.Event()
        self.stop_event = asyncio.Event()        
        print ( self.hparams )

    async def run(self):
        # Main loop.
        self.loop = asyncio.get_running_loop()
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()
        while True:
            
            try:
                # Start step.
                logger.info('\n' + '-' * 40 + ' Step ' + '-' * 40)
                logger.info(f"Step: {self.global_step}, Mask: {self.current_window}, "
                            f"Block: {self.current_block}, Time: {int(time.time())}")
                global_step_start_time = time.time()
                self.global_step += 1

                # Download files.    
                logger.info(f"\tDownloading masks for: {[self.current_window-1, self.current_window]}")
                start_time = time.time()
                files = await download_files_for_buckets_and_masks(buckets=self.buckets, masks=[self.current_window-1, self.current_window])
                logger.info(f"\t\tDownloaded {sum([len(files[k]) for k in files])} masks for: {[self.current_window-1, self.current_window]} "
                            f"in {time.time() - start_time} seconds")
                
                # Apply masks from previous window.
                logger.info(f"\tLoading masks from: {self.current_window - 1}")
                start_time = time.time()
                indices = await get_indices_for_mask(self.model, self.current_window - 1, self.hparams.compression)
                mask_files = await get_files_for_mask_from_temp(mask = self.current_window - 1)
                logger.info(f"\t\tLoaded {len(mask_files)} masks for: {self.current_window - 1} "
                            f"in {time.time() - start_time} seconds")

                # Apply masks to model and average them.
                logger.info(f"\tApplying {len(mask_files)} masks to model.")
                start_time = time.time()
                masks_per_param = {name: 0 for name, _ in self.model.named_parameters()}
                for file_i in mask_files:
                    try:
                        mask = torch.load(file_i, map_location=torch.device(self.model.device), weights_only=True)
                        for name, param in self.model.named_parameters():
                            if name not in indices or name not in mask:
                                continue
                            values = mask[name].to(self.model.device)
                            indices_name = indices[name].to(self.model.device)
                            param.data.view(-1)[indices_name] += values
                            masks_per_param[name] += 1
                            del values
                    except Exception:
                        logger.exception(f"Error applying mask from {file_i}")
                logger.info(f"\t\tApplied {len(mask_files)} masks to model in {time.time() - start_time} seconds")

                # Average them on the model.
                logger.info(f"\tAveraging masks on model.")
                for name, param in self.model.named_parameters():
                    if name not in masks_per_param or name not in indices or masks_per_param[name] == 0:
                        continue
                    indices_name = indices[name].to(self.config.device)
                    param.data.view(-1)[indices_name] /= (masks_per_param[name] + 1)
                logger.info(f"\t\tAveraged masks on model in {time.time() - start_time} seconds")
                
                # Train for the current mask.
                logger.info(f"\tLoading {self.optimal_pages_per_step} page dataset")
                start_time = time.time()
                start_mask = self.current_window
                pages = SubsetFineWebEdu2Loader.next_pages(
                    offset=self.current_block * self.hparams.pages_window_speed,
                    n_pages = self.optimal_pages_per_step,
                    seed=self.uid
                )
                dataset = SubsetFineWebEdu2Loader(
                    batch_size=self.config.actual_batch_size,
                    sequence_length=self.hparams.sequence_length,
                    pages_info=pages,
                    tokenizer=self.hparams.tokenizer
                )
                logger.info(f"\t\tLoaded dataset pages: {[p[1] for p in pages]} in {time.time() - start_time} seconds")

                # Train the model on the current page.
                logger.info(f"\tTraining on pages: {[p[1] for p in pages]}")
                start_time = time.time()
                torch.cuda.empty_cache()  # Empty cache going into the training step.
                self.optimizer.zero_grad()  # Clear any lingering grads.
                total_loss = 0.0
                total_steps = self.hparams.desired_batch_size // self.config.actual_batch_size
                for idx, batch in enumerate(dataset):
                    input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                    labels = input_ids.clone()
                    labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                    with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):  # Enable autocasting
                        outputs = self.model(input_ids=input_ids, labels=labels)
                    total_loss += outputs.loss.item()
                    loss = outputs.loss / (total_steps + 1)  # Divide by number of accumulations.
                    loss.backward()
                    # Break on total steps or if we have a new mask.
                    if idx >= total_steps - 1:
                        logger.info('\t\tBreak training, no more steps.')
                        self.optimal_pages_per_step += 1
                        break
                    elif start_mask != self.current_window:
                        logger.info(f'\t\tBreak training, new mask {start_mask} != {self.current_window}')
                        self.optimal_pages_per_step = max(1, self.optimal_pages_per_step - 1)
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
                average_loss = total_loss / (idx + 1)
                total_time = time.time() - start_time
                steps_per_second = (idx + 1) / total_time
                batches_per_second = self.config.actual_batch_size * (idx + 1) / total_time
                tokens_per_second = self.hparams.sequence_length * self.config.actual_batch_size * (idx + 1) / total_time
                if self.config.use_wandb:
                    wandb.log({
                        "step_loss": average_loss,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "incentive": float(self.metagraph.I[self.uid]),
                        "tokens_per_second": tokens_per_second
                    })
                logger.info(f"\t\tLoss: {average_loss}, learning_rate: {self.scheduler.get_last_lr()[0]}")
                logger.info(f"\t\tTraining completed in {total_time} seconds, Steps per second: {steps_per_second}, "
                            f"Batches per second: {batches_per_second}, Tokens per second: {tokens_per_second}")

                # Upload our model mask to S3.
                logger.info(f"\tUploading for mask: {self.current_window}")
                start_time = time.time()
                await upload_mask(self.config.bucket, self.model, self.current_window, self.wallet, self.hparams.compression)
                logger.info(f"\t\tFinished upload for mask: {self.current_window} in {time.time() - start_time} seconds.")
                
                # Delete lingering files 
                logger.info(f"\tCleaning space.")
                start_time = time.time()
                await delete_files_before_mask( mask_max = self.current_window - self.hparams.max_history )
                await delete_files_from_bucket_before_mask( bucket = self.config.bucket, mask_max = self.current_window - self.hparams.max_history )
                logger.info(f"\t\tFinished cleaning space in {time.time() - start_time} seconds.")

                # Calculate and log global steps per second
                global_step_total_time = time.time() - global_step_start_time
                global_steps_per_second = 1 / global_step_total_time
                if self.config.use_wandb:
                    wandb.log({
                        "global_steps_per_second": global_steps_per_second,
                        "global_step_time": global_step_total_time,
                        "global_tokens_per_second": tokens_per_second / global_step_total_time 
                    })
                print (f'\nGlobal step completed in {global_step_total_time} seconds\n')
                
            # Catch keyboard interrrupt.
            except KeyboardInterrupt:
                logger.info("Training interrupted by user. Stopping the run.")
                sys.exit(0)
            
            # Catch unknown.
            except Exception as e:
                logger.exception(f"Exception during training loop: {e}")
                continue

    # Returns the mask window based on a block.
    def block_to_window(self, block: int) -> int:
        return int(block / self.hparams.mask_window_length)

    # A listener thread which posts the block event
    # when the chain announces a new block.
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            loop.call_soon_threadsafe(self.block_event.set)
            if self.block_to_window(self.current_block) != self.current_window:
                self.current_window = self.block_to_window(self.current_block)
                loop.call_soon_threadsafe(self.new_window_event.set)
                logger.info(f"\t\tNew mask: {self.current_window}")
        # Subscribe to block headers with the custom handler
        bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)

    # Helper for waiting on block time
    async def wait_for_new_block(self):
        while True:
            await self.block_event.wait()
            self.block_event.clear()

    # Helper for waiting on mask time.
    async def wait_for_new_mask(self):
        while True:
            await self.new_window_event.wait()
            self.new_window_event.clear()
            
if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.run())
