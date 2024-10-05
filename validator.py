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
        self.scores = torch.zeros( 256, dtype = torch.float32 ) 
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
                    self.scores[idx] = 0                            # Reset rewards for the changed hotkey
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
                slice_infos = (await download_slices_for_buckets_and_windows(
                    buckets=self.buckets,
                    windows=[self.eval_window]
                ))[self.eval_window]
                logger.info(f"\t\tDownloaded {len(slice_infos)} slices for previous window: {self.eval_window} in {time.time() - start_time} seconds")

                # If there are no slices to eval, wait until the next window then start again.
                if len(slice_infos) == 0:
                    while self.current_window == self.step_window:
                        await asyncio.sleep(0.1)
                    continue

                # Step 2: Apply slices to the model from the previous window.
                logger.info(f"\tApplying slices from previous window: {self.eval_window} to model.")
                start_time = time.time()
                eval_slices = await apply_slices_to_model(
                    model=self.model,
                    window=self.eval_window,  # Get files from previous window.
                    seed=self.window_seeds[self.step_window],  # Use seed as the hash of the current window.
                    compression=self.hparams.compression
                )
                applied_per_step = len(eval_slices)
                logger.info(f"\t\tApplied {applied_per_step} slices from previous window: {self.eval_window} with seed: {self.window_seeds[self.step_window]} in {time.time() - start_time} seconds")

                # Step 3: Load the slices and pages for each.
                logger.info(f"\tLoading slices from previous window: {self.eval_window}")
                uids_to_eval = []
                slices_to_eval = []
                pages_to_eval = []
                start_time = time.time()
                for info_i in slice_infos:
                    try:
                        uid = self.metagraph.hotkeys.index(info_i.hotkey)
                    except ValueError:
                        logger.warning(f"Hotkey {info_i.hotkey} not found in metagraph hotkeys.")
                        continue
                    uids_to_eval.append(uid)
                    # Load the slice for the current miner.
                    slice_data = torch.load(
                        info_i.temp_file,
                        map_location=torch.device(self.model.device),
                        weights_only=True
                    )
                    slices_to_eval.append(slice_data)
                    # Determine the offset for page loading.
                    offset_i = self.eval_window * self.hparams.window_length * self.hparams.window_speed
                    # Sample pages for evaluation.
                    sampled_pages = random.sample(
                        await AsyncSubsetFineWebEdu2Loader.next_pages(
                            offset=offset_i,
                            n_pages=self.hparams.validator_window_eval_size,
                            seed=uid
                        ),
                        self.hparams.validator_pages_per_eval
                    )
                    pages_to_eval.extend(sampled_pages)
                logger.info(f"\t\tLoaded {len(slices_to_eval)} slices for uids: {uids_to_eval} from previous window: {self.eval_window} in {time.time() - start_time} seconds")

                # Step 4: Load the full eval dataset from all the pages.
                logger.info(f"\tDownloading {len(pages_to_eval)} pages for uids: {uids_to_eval}")
                random.shuffle(pages_to_eval)  # Shuffle to ensure randomness in evaluation.
                start_time = time.time()
                eval_dataset = await AsyncSubsetFineWebEdu2Loader.create(
                    batch_size=self.config.actual_batch_size,
                    sequence_length=self.hparams.sequence_length,
                    pages_info=pages_to_eval,
                    tokenizer=self.hparams.tokenizer
                )
                logger.info(f"\t\tFinished downloading {len(pages_to_eval)} pages in {time.time() - start_time} seconds")

                # Step 5: Compute gradients on this dataset.
                logger.info(f"\tComputing gradients for {len(pages_to_eval)} pages")
                start_time = time.time()
                self.model.zero_grad()
                self.model.eval()
                # Enable gradient computation
                with torch.enable_grad():
                    for batch in eval_dataset:
                        input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss
                        loss.backward()
                        # Check if a new window has started to possibly abort early.
                        if self.current_window != self.step_window:
                            logger.info(f'\t\tBreak eval on new window: {self.current_window} > {self.step_window}')
                            break
                logger.info(f"\t\tFinished computing gradients in {time.time() - start_time} seconds")

                # Step 6: Compute slice importance using second-order approximation with Fisher Information Matrix.
                logger.info(f"\tComputing slice importance scores using second-order approximation")
                start_time = time.time()
                step_scores = torch.zeros(256, dtype=torch.float32).to(self.model.device)  # Initialize step scores.
                indices = await get_indices_for_window(
                    model=self.model,
                    seed=self.window_to_seed(self.eval_window + 1),  # Seed index for the eval window.
                    compression=self.hparams.compression
                )

                # Collect gradients for all parameters.
                gradients = {}
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue
                    gradients[name] = param.grad.view(-1).clone().detach()

                # Compute Fisher Information Matrix diagonal approximation (g_i^2).
                fisher_diagonal = {}
                for name, grad in gradients.items():
                    fisher_diagonal[name] = grad.pow(2)

                # Iterate over each slice to compute importance scores.
                for idx, slice_i in enumerate(slices_to_eval):
                    uid = uids_to_eval[idx]
                    delta_L = 0.0  # Initialize importance score for the current slice.
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        # Retrieve the indices for the current parameter.
                        param_indices = indices[name].to(self.model.device)
                        # Flatten the parameter gradients and slices.
                        g = gradients[name][param_indices]  # Gradient vector.
                        s = slice_i[name].view(-1).to(self.model.device)  # Slice vector.
                        # Compute the first-order term: -g^T s
                        first_order = -torch.dot(g, s).item()
                        # Compute the second-order term: 0.5 * s^T H s, where H ≈ diag(g^2)
                        second_order = 0.5 * torch.dot(s.pow(2), fisher_diagonal[name][param_indices]).item()
                        # Accumulate the importance score.
                        delta_L += first_order + second_order
                    # Assign the computed importance score to the corresponding UID.
                    step_scores[uid] += delta_L
                logger.info(f"\t\tFinished computing slice importance scores in {time.time() - start_time} seconds")

                # Step 7: Normalize the scores as rewards and use them as weights.
                logger.info(f"\tNormalizing scores to compute weights")
                start_time = time.time()
                # Update moving average of scores.
                self.scores = self.hparams.validator_moving_alpha * step_scores.cpu() + (1 - self.hparams.validator_moving_alpha) * self.scores
                # Apply softmax to normalize scores into weights.
                # To ensure numerical stability, subtract the max before exponentiating.
                if torch.any(self.scores != 0):
                    scores_nonzero = self.scores[self.scores != 0]
                    max_score = torch.max(scores_nonzero)
                    normalized_scores = torch.softmax((scores_nonzero - max_score) * self.hparams.validator_weights_temperature, dim=0)
                    self.weights[self.scores != 0] = normalized_scores
                logger.info(f"\t\tNormalized scores into weights in {time.time() - start_time} seconds")

                # Step 8: Log the results if using Weights and Biases.
                if self.config.use_wandb:
                    for uid in torch.nonzero(self.weights).squeeze():
                        wandb.log({
                            f"step_scores/{uid.item()}": step_scores[uid].item(),
                            f"moving_scores/{uid.item()}": self.scores[uid].item(),
                            f"weights/{uid.item()}": self.weights[uid].item(),
                        })

                # Step 9: Log the rewards, weights, and scores.
                logger.info('\tWeights:')
                nonzero_weights = torch.nonzero(self.weights).squeeze()
                for uid, moving_score, weight, step_score in zip(
                    nonzero_weights,
                    self.scores[nonzero_weights],
                    self.weights[nonzero_weights],
                    step_scores[nonzero_weights]
                ):
                    logger.info(f"\t\tuid: {uid.item()}, step_score: {step_score.item():.6f}, moving_score: {moving_score.item():.6f}, weight: {weight.item():.6f}")
     
                # Delete lingering files 
                logger.info(f"\tCleaning space.")
                start_time = time.time()
                del slices_to_eval
                del gradients
                del fisher_diagonal
                torch.cuda.empty_cache()
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
