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
        parser.add_argument('--project', type=str, default='OMINU', help='Optional wandb project name')
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
        logger.info('[bold]' + '\n' + '-' * 40 + ' Config ' + '-' * 40)
        logger.info(self.config)

        # Init bittensor objects.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise ValueError(f'Wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}')
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info('[bold]' + '\n' + '-' * 40 + ' Objects ' + '-' * 40)
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
        logger.info('[bold]' + '\n' + '-' * 40 + ' Hparams ' + '-' * 40)
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
        self.step_score = torch.zeros( 256, dtype = torch.float32 ) 
        self.scores = torch.zeros( 256, dtype = torch.float32 ) 
        self.weights = torch.zeros( 256, dtype = torch.float32 ) 
        self.sample_rate = 0.1
        logger.info( self.hparams )
        
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
                # Start step.
                logger.info('[bold]' + '\n' + '-' * 40 + f' Step: [chartreuse4]{self.global_step}[/chartreuse4] ' + '-' * 40)
                step_start_time = time.time()
                self.global_step += 1
                self.step_window = self.current_window
                self.eval_window = self.current_window - 3
                logger.info(f"Step: [chartreuse4]{self.global_step}[/chartreuse4], Step Window: [wheat4]{self.step_window}[/wheat4], Eval Window: [wheat4]{self.eval_window}[/wheat4]"
                            f" Block: [wheat4]{self.current_block}[/wheat4], Time: [cadet_blue]{int(step_start_time)}[/cadet_blue]")
                
                # Download the slices for the window.
                logger.info(f"\t[cornflower_blue]Downloading[/cornflower_blue] slices from previous window: [wheat4]{ self.eval_window }[/wheat4]")
                start_time = time.time()
                slice_infos = await download_slices_for_buckets_and_windows(
                    buckets=self.buckets,
                    windows=[self.eval_window]
                )
                # If there are no slices to eval, wait until the next window then start again.
                if self.eval_window not in slice_infos or len(slice_infos[self.eval_window]) == 0:
                    print ('\t\tNo slices to download, waiting for next window...')
                    while self.current_window == self.step_window: await asyncio.sleep(0.1)
                    continue
                slice_infos = slice_infos[self.eval_window]
                logger.info(f"\t\tDownloaded {len(slice_infos)} slices for previous window: [wheat4]{self.eval_window}[/wheat4] in [cadet_blue]{time.time() - start_time}[/cadet_blue] seconds")
                
                indices = await get_indices_for_window(
                    model=self.model,
                    seed=self.window_to_seed(self.eval_window + 1),  # Seed index for the eval window.
                    compression=self.hparams.compression
                )       
                
                # Step 2: Compute slice importance using second-order approximation with Fisher Information Matrix.
                eval_start_time = time.time()         
                info_i = random.choice(slice_infos)
                
                # Get the UID we are evalling.
                try: eval_uid = self.metagraph.hotkeys.index(info_i.hotkey)
                except ValueError:
                    logger.warning(f"Hotkey [yellow]{info_i.hotkey}[/yellow] not found in metagraph hotkeys.")
                    continue
                
                # Load the slice for the current miner.
                logger.info(f"\t[cornflower_blue]Loading[/cornflower_blue] slice from hotkey: [yellow]{info_i.hotkey}[/yellow] and uid: [light_slate_blue]{eval_uid}[/light_slate_blue]")
                start_time = time.time()
                slice_data = await load_slices( filename = info_i.temp_file, device = self.model.device )
                logger.info(f"\t\tLoaded slice in [cadet_blue]{time.time() - start_time}[/cadet_blue] seconds")

                # Load the dataset for this miner.
                start_time = time.time()
                offset_i = self.eval_window * self.hparams.window_length * self.hparams.window_speed
                seed = eval_uid
                sampled_pages = await AsyncSubsetFineWebEdu2Loader.next_pages(
                    offset = offset_i,
                    n_pages = self.hparams.validator_window_eval_size,
                    seed = seed
                )
                random.shuffle(sampled_pages) # Important to not preference early pages.
                logger.info(f"\t[cornflower_blue]Getting[/cornflower_blue] pages: [medium_orchid3]{[p[1] for p in sampled_pages]}[/medium_orchid3] for offset: [wheat4]{offset_i}[/wheat4], uid: [light_slate_blue]{eval_uid}[/light_slate_blue] and seed: [yellow]{seed}[/yellow]")
                eval_dataset = await AsyncSubsetFineWebEdu2Loader.create(
                    batch_size=self.config.actual_batch_size,
                    sequence_length=self.hparams.sequence_length,
                    pages_info = sampled_pages,
                    tokenizer=self.hparams.tokenizer
                )
                logger.info(f"\t\tLoaded pages in [cadet_blue]{time.time() - start_time}[/cadet_blue] seconds")
                
                # Run the eval.
                start_time = time.time()
                self.model.zero_grad()
                self.model.eval()                
                logger.info(f"\t[cornflower_blue]Evalling[/cornflower_blue] without slice from: [light_slate_blue]{eval_uid}[/light_slate_blue] with sample rate: [turquoise4]{self.sample_rate}[/turquoise4]")
                full_steps = 0
                without_loss = 0
                eval_idxs = set()
                with torch.no_grad():
                    for idx, batch in enumerate(eval_dataset):
                        # Randomly sample every sample_rate examples
                        if random.random() < self.sample_rate:
                            eval_idxs.add( idx )
                            full_steps += 1
                            input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                            labels = input_ids.clone()
                            labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                            with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):  # Enable autocasting
                                outputs = self.model(input_ids=input_ids, labels=labels)
                            without_loss += outputs.loss.item()
                logger.info(f"\t\tFinished Steps: [turquoise4]{idx}[/turquoise4], Applied: [turquoise4]{full_steps}[/turquoise4], Rate: [turquoise4]{full_steps/(idx + 1)}[/turquoise4], Sample Probability: [turquoise4]{self.sample_rate}[/turquoise4] in [cadet_blue]{time.time() - start_time}[/cadet_blue] seconds")

                # Apply slice to model.   
                logger.info(f"\t[cornflower_blue]Applying[/cornflower_blue] slice to model.")
                start_time = time.time()
                for name, param in self.model.named_parameters():
                    if name not in indices or name not in slice_data:
                        continue
                    values = slice_data[name].to(self.model.device)
                    param_indices = indices[name].to(self.model.device)
                    param.data.view(-1)[param_indices] = values
                    del values
                logger.info(f"\t\tSlice applied in [cadet_blue]{time.time() - start_time}[/cadet_blue] seconds")
                            
                # Recompute the loss.
                logger.info(f"\t[cornflower_blue]Evalling[/cornflower_blue] with slice from: [light_slate_blue]{eval_uid}[/light_slate_blue] with sample rate: [turquoise4]{self.sample_rate}[/turquoise4]")
                with_loss = 0
                full_steps = 0
                start_time = time.time()
                with torch.no_grad():
                    for idx, batch in enumerate(eval_dataset):
                        # Randomly sample every sample_rate examples
                        if idx in eval_idxs:
                            full_steps += 1
                            input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                            labels = input_ids.clone()
                            labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                            with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):  # Enable autocasting
                                outputs = self.model(input_ids=input_ids, labels=labels)
                            with_loss += outputs.loss.item()
                logger.info(f"\t\tFinished Steps: [turquoise4]{idx}[/turquoise4], Applied: [turquoise4]{full_steps}[/turquoise4], Rate: [turquoise4]{full_steps/(idx + 1)}[/turquoise4], Sample Probability: [turquoise4]{self.sample_rate}[/turquoise4] in [cadet_blue]{time.time() - start_time}[/cadet_blue] seconds")
                            
                # Check to see if we exhuasted the window.
                if self.step_window != self.current_window:
                    self.sample_rate = max(0.0001, self.sample_rate * 0.99)
                else:
                    self.sample_rate = min(1, self.sample_rate * 1.01)
           
                # # Clean up GPU memory
                torch.cuda.empty_cache()
    
                # Step 7: Normalize the scores as rewards and use them as weights.
                start_time = time.time()
                self.step_score[eval_uid] = with_loss - without_loss
                logger.info(f'\t\tStep score for uid: [light_slate_blue]{eval_uid}[/light_slate_blue] of [aquamarine1]{self.step_score[eval_uid]}[/aquamarine1]')
                logger.info('\t[cornflower_blue]Weights[/cornflower_blue]:')
                self.scores[eval_uid] = (1 - self.hparams.validator_moving_alpha) * self.step_score[eval_uid] + self.hparams.validator_moving_alpha * self.scores[eval_uid]
                self.scores[torch.isnan(self.scores)] = 0
                valid_score_indices = torch.nonzero((self.scores != 0) & (~torch.isnan(self.scores))).squeeze().view(-1, 1)
                valid_scores = self.scores[valid_score_indices].view(-1, 1) if valid_score_indices.dim() == 0 else self.scores[valid_score_indices]
                print (self.scores)
                if len(valid_scores) > 0:
                    max_score = torch.max(valid_scores)
                    normalized_scores = torch.softmax((valid_scores - max_score) * self.hparams.validator_weights_temperature, dim=0)
                    self.weights[valid_score_indices] = normalized_scores
                if self.config.use_wandb:
                    for uid_i in valid_score_indices:
                        wandb.log({
                            f"step_score/{uid_i.item()}": self.step_score[uid_i].item(),
                            f"moving_scores/{uid_i.item()}": self.scores[uid_i].item(),
                            f"weights/{uid_i.item()}": self.weights[uid_i].item(),
                            'self.sample_rate': self.sample_rate,
                        })
                for uid_i in valid_score_indices:
                    moving_score = self.scores[uid_i].item()
                    weight = self.weights[uid_i].item()
                    step_score = self.step_score[uid_i].item()
                    logger.info(f"\t\tuid: [light_slate_blue]{uid_i.item()}[/light_slate_blue], step_score: [aquamarine1]{step_score:.6f}[/aquamarine1], moving_score: [aquamarine1]{moving_score:.6f}[/aquamarine1], weight: [aquamarine1]{weight:.6f}[/aquamarine1]")

                # Delete lingering files 
                logger.info(f"\t[cornflower_blue]Cleaning[/cornflower_blue] space.")
                start_time = time.time()
                await delete_files_before_window( window_max = self.current_window - self.hparams.max_history )
                logger.info(f"\t\tFinished cleaning space in [cadet_blue]{time.time() - start_time}[/cadet_blue] seconds.")
                
                # Step 2: Apply slices to the model from the previous window.
                logger.info(f"\t[cornflower_blue]Applying[/cornflower_blue] slices from previous window: [wheat4]{self.eval_window}[/wheat4] to model.")
                start_time = time.time()
                eval_slices = await apply_slices_to_model(
                    model=self.model,
                    window=self.eval_window,  # Get files from previous window.
                    seed=self.window_seeds[self.step_window],  # Use seed as the hash of the current window.
                    compression=self.hparams.compression
                )
                applied_per_step = len(eval_slices)
                logger.info(f"\t\tApplied {applied_per_step} slices from previous window: [wheat4]{self.eval_window}[/wheat4] with index seed: [yellow]{self.window_seeds[self.step_window]}[/yellow] in [cadet_blue]{time.time() - start_time}[/cadet_blue] seconds")
                
                # Ensure window is over.
                logger.info( f'\n[bold white]Global step completed in [cadet_blue]{time.time() - step_start_time}[/cadet_blue] seconds\n[/bold white]')
                while self.current_window == self.step_window: await asyncio.sleep(0.1)
                                        
                # Set temperatured weights on the chain.
                # TODO(const): make this work.
                # if self.current_window % 100 == 0: 
                #     self.subtensor.set_weights(
                #         wallet = self.wallet,
                #         netuid = self.config.netuid,
                #         uids = self.metagraph.uids,
                #         weights = self.weights[ self.metagraph.uids ],
                #         wait_for_inclusion = False,
                #         wait_for_finalization = False,
                #     )
                        
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
