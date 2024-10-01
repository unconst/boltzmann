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
                
                # Eval until mask changes.
                eval_window = self.current_window - 1
                while start_mask != self.current_window:
                    logger.info(f"\tEval Step.")

                    # Get random UID to eval.
                    miner_uid = random.choice( list(metagraph.uids) )
                    logger.info(f"\tEvalling {miner_uid}")

                    miner_filename = os.path.join(tempfile.gettempdir(), f"mask-{eval_window}-{metagraph.hotkeys[miner_uid]}")
                    logger.info(f"\tMask filename: {miner_filename}")
                    
                    mask_indicies = await get_files_for_mask_from_temp (mask = eval_window )
                    logger.info(f"\tLoaded window mask.")

                    # Train for the current mask.
                    pages = SubsetFineWebEdu2Loader.next_pages(
                        offset = self.current_block * self.hparams.pages_window_speed,
                        n_pages = 1,
                        seed = miner_uid
                    )
                    dataset = SubsetFineWebEdu2Loader(
                        batch_size = self.config.actual_batch_size,
                        sequence_length = self.hparams.sequence_length,
                        pages_info = pages,
                        tokenizer = self.hparams.tokenizer
                    )
                    logger.info(f"\tLoaded eval pages: {[p[1] for p in pages]} dataset for offset: {self.current_block * self.hparams.pages_window_speed} ")
                    
                    # Zero grad on the model and compute loss.
                    model.zero_grad()
                    for idx, batch in enumerate(dataset):
                        input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                        with torch.amp.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                            outputs = model(input_ids=input_ids, labels=labels)
                            loss = outputs.loss
                            loss.backward()  # Compute gradients
                    print(f"\tComputed gradient on model for pages")

                    # Collect the gradients
                    gradients = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            gradients[name] = param.grad.detach().clone()
                        else:
                            # If the parameter did not receive a gradient, we set it to zero
                            gradients[name] = torch.zeros_like(param.data)
                    print(f"\tCollected gradients.")

                    # Step 4: Flatten the gradients and the miner's update (mask values)
                    step_start_time = time.time()
                    gradient_vector = []
                    update_vector = []
                    for name in sorted(model.state_dict().keys()):
                        # Ensure we're working with parameters that have gradients and updates
                        if name in gradients and name in mask_values:
                            grad = gradients[name].view(-1)
                            # Initialize a zero vector for the parameter
                            update = torch.zeros_like(grad)
                            # Get the indices and values of the miner's update (mask)
                            indices = mask_indices[name].to(model.device)
                            values = mask_values[name].to(model.device)
                            # Place the values at the correct indices
                            update[indices] = values
                            # Append to the vectors
                            gradient_vector.append(grad)
                            update_vector.append(update)
                        else:
                            # If no gradient or update, append zeros
                            size = model.state_dict()[name].numel()
                            gradient_vector.append(torch.zeros(size, device=model.device))
                            update_vector.append(torch.zeros(size, device=model.device))
                    # Concatenate all parameter gradients and updates into single tensors
                    gradient_vector = torch.cat(gradient_vector)
                    update_vector = torch.cat(update_vector)
                    step_end_time = time.time()
                    print(f"\tFlatten gradients.")

                    # Compute reward.
                    dot_product = torch.dot(gradient_vector, update_vector)
                    lambda_reg = hparams.update_norm_regularization  # Regularization coefficient; adjust as needed
                    update_norm = torch.norm(update_vector)
                    regularization = lambda_reg * update_norm.item()
                    reward = max(0.0, -dot_product.item() - regularization)
                    weights[miner_uid] = (reward * hparams.weights_alpha) + ((1 - hparams.weights_alpha) * weights[miner_uid])
                    if config.use_wandb:
                        wandb.log({f"R/{miner_uid}": reward, f"W/{miner_uid}": weights[miner_uid]})
                    print(f'\tupdate_norm: {update_norm}')
                    print(f'\tregularization: {regularization}')
                    print(f'\tdot_product: {dot_product}')
                    print(f'\treward: {reward}')
                    print(f'\tweights: {weights[miner_uid]}')
                                        
                    # Clean up to free memory
                    del gradients
                    del gradient_vector
                    del update_vector
                    del mask_indices
                    del mask_values
                    os.remove(mask_temp_file)      
                # We can't download the mask for the miner.    
                except Exception as e:
                    print(f"Miner eval failed with error: {e}, setting score of zero.")
                    weights[ miner_uid ] = ( 0.0 * hparams.weights_alpha ) + ( (1 - hparams.weights_alpha) * weights[ miner_uid ] )

                # Delete lingering files 
                logger.info(f"\tCleaning space.")
                start_time = time.time()
                await delete_files_before_mask( mask_max = self.current_window - self.hparams.max_history )
                logger.info(f"\t\tFinished cleaning space in {time.time() - start_time} seconds.")
                
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
