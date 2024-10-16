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
from typing import List, Dict, Any
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
    """Miner class for training models in a decentralized network.

    This class handles model training, state synchronization, and communication
    with the Bittensor blockchain, supporting multi-GPU training.

    Attributes:
        config (bt.Config): Configuration object with parameters.
        wallet (bt.Wallet): Wallet object for blockchain interactions.
        subtensor (bt.Subtensor): Subtensor object for accessing the blockchain.
        metagraph (bt.Metagraph): Metagraph object representing the network.
        uid (int): Unique identifier of the miner in the network.
        buckets (List[str]): List of buckets (storage locations) in the network.
        num_gpus (int): Number of GPUs to use for training.
        model (torch.nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        hparams (Namespace): Hyperparameters for model training.
    """

    @staticmethod
    def config() -> bt.Config:
        """Parse command-line arguments and initialize the configuration.

        Returns:
            bt.Config: The configuration object with all parameters.
        """
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
        parser.add_argument('--random', action='store_true', help='Train on random pages')
        parser.add_argument('--sync_state', action='store_true', help='Sync the model state by pulling from the history.')
        parser.add_argument('--baseline', action='store_true', help='Do not perform syncing with other peers, just train.')
        parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for training (default: 1)')
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        config = bt.config(parser)
        config.subtensor.network = 'test'
        config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
        if config.debug: debug()
        if config.trace: trace()
        return config

    def __init__(self):
        # Initialize configuration.
        self.config = Miner.config()
        logger.info('\n' + '-' * 40 + ' Config ' + '-' * 40)
        logger.info(self.config)

        # Initialize blockchain objects.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise ValueError(f'Wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}')
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info('\n' + '-' * 40 + ' Objects ' + '-' * 40)
        logger.info(f'Wallet: {self.wallet}\nSubtensor: {self.subtensor}\nMetagraph: {self.metagraph}\nUID: {self.uid}')

        # Initialize bucket.
        try:
            if self.config.bucket != self.subtensor.get_commitment(self.config.netuid, self.uid):
                raise ValueError('Bucket mismatch.')
        except:
            self.subtensor.commit(self.wallet, self.config.netuid, self.config.bucket)
        logger.info(f'Bucket: {self.config.bucket}')

        # Initialize WandB.
        if self.config.use_wandb:
            try:
                for run in wandb.Api().runs(path=self.config.project):
                    if run.name == f'M{self.uid}':
                        logger.info(f'Deleting old run: {run}')
                        run.delete()
            except Exception as e:
                logger.warning(f'Failed to delete old runs: {e}')
            wandb.init(project=self.config.project, resume='allow', name=f'M{self.uid}', config=self.config)

        # Set number of GPUs.
        self.num_gpus = min(torch.cuda.device_count(), self.config.num_gpus)

        # Initialize model.
        logger.info('\n' + '-' * 40 + ' Hparams ' + '-' * 40)
        self.hparams = load_hparams()
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model.to(self.config.device)
        self.model.train()

        # Initialize optimizer and scheduler.
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000)

        # Initialize other variables.
        self.global_step = 0
        self.sample_rate = 1.0

        # Initialize bucket list.
        self.buckets = [self.config.bucket]
        if self.config.remote:
            self.buckets.extend(self.metagraph.buckets)

        # Initialize events and threading.
        self.new_block_event = asyncio.Event()
        self.new_window_event = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.get_current_block()
        self.current_window = self.block_to_window(self.current_block)
        self.window_time = T()
        self.window_duration = 0.0
        self.window_seeds = {}
        self.loop = asyncio.get_event_loop()
        self.block_thread = threading.Thread(target=self.block_listener, args=(self.loop,))
        self.block_thread.start()

    async def train_on_slice(self, slice_idx: int):
        """Train on a specific slice using the assigned GPU.

        Args:
            slice_idx (int): Index of the slice, corresponding to the GPU index.
        """
        device = torch.device(f'cuda:{slice_idx}')
        torch.cuda.set_device(device)

        # Determine the seed for this slice.
        if self.num_gpus == 1:
            # Single GPU scenario: use original seed.
            seed = self.window_to_seed(self.current_window)
        else:
            # Multi-GPU scenario: use different seeds for each slice.
            seeds = self.window_to_seeds(self.current_window, self.num_gpus)
            seed = seeds[slice_idx]

        # Load dataset for this slice.
        pages = await DatasetLoader.next_pages_async(
            offset=self.current_window,
            n_pages=self.hparams.validator_window_eval_size,
            seed=str(seed),
            num_rows_per_page=self.hparams.num_rows_per_page
        )

        dataset = await DatasetLoader.create(
            batch_size=self.config.actual_batch_size,
            sequence_length=self.hparams.sequence_length,
            pages_info=pages,
            tokenizer=self.hparams.tokenizer
        )

        # Create a separate model instance for this slice to avoid conflicts.
        slice_model = LlamaForCausalLM(config=self.hparams.model_config)
        slice_model.load_state_dict(self.model.state_dict())
        slice_model.to(device)
        slice_model.train()

        # Initialize optimizer and scheduler for this slice.
        optimizer = optim.AdamW(slice_model.parameters(), lr=self.hparams.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000)

        # Training loop for this slice.
        total_loss = 0.0
        full_steps = 0
        train_start = T()
        for batch in dataset:
            input_ids = torch.tensor(batch, dtype=torch.long).to(device)
            labels = input_ids.clone()
            labels[labels == self.hparams.tokenizer.pad_token_id] = -100

            outputs = slice_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            full_steps += 1

        train_duration = T() - train_start
        avg_loss = total_loss / (full_steps + 1)

        # Calculate tokens processed.
        tokens_per_step = self.hparams.sequence_length * self.config.actual_batch_size * (full_steps + 1)
        tokens_per_second = tokens_per_step / train_duration

        logger.info(
            f"Slice {slice_idx}: Trained with avg loss {avg_loss:.4f}, "
            f"tokens per step {tokens_per_step}, tokens per second {tokens_per_second:.2f}"
        )

        # Compute the delta between the initial and updated model parameters.
        delta = {}
        with torch.no_grad():
            for name, param in slice_model.named_parameters():
                delta[name] = param.data.cpu() - self.model.state_dict()[name].cpu()

        # Save the computed delta for this slice.
        await save_slice_for_window(
            model_state=delta,
            window=self.current_window,
            seed=seed,
            compression=self.hparams.compression,
            key=f'delta_{slice_idx}'
        )

        # Clean up.
        del slice_model
        torch.cuda.empty_cache()

    async def run(self):
        """Main async function to run the miner."""
        self.loop = asyncio.get_running_loop()
        self.update_task = asyncio.create_task(self.update_weights())

        # Load initial model state if syncing is enabled.
        if self.config.sync_state and not self.config.baseline:
            window = self.current_window
            st = T()
            state_slices = await download_slices_for_buckets_and_windows(
                buckets=self.buckets,
                windows=[window],
                key='state'
            )
            if state_slices:
                await apply_slices_to_model(
                    model=self.model,
                    state_slices=state_slices,
                    compression=self.hparams.compression
                )
                logger.info(f"{P(window, T() - st)}: Synced model state for window {window}.")
            else:
                logger.warning(f"No state slices found for window {window}.")

        # Main training loop.
        while True:
            try:
                # Get the current window.
                start_step = T()
                window = self.current_window

                # Download state slices from other miners.
                if not self.config.baseline:
                    st = T()
                    state_slices = await download_slices_for_buckets_and_windows(
                        buckets=self.buckets,
                        windows=[window],
                        key='state'
                    )
                    if state_slices:
                        await apply_slices_to_model(
                            model=self.model,
                            state_slices=state_slices,
                            compression=self.hparams.compression
                        )
                        logger.info(f"{P(window, T() - st)}: Downloaded and applied state slices for window {window}.")
                    else:
                        logger.warning(f"No state slices found for window {window}.")

                # Start training tasks for each slice (GPU).
                training_tasks = []
                for i in range(self.num_gpus):
                    task = asyncio.create_task(self.train_on_slice(slice_idx=i))
                    training_tasks.append(task)
                await asyncio.gather(*training_tasks)

                # Aggregate deltas from all slices.
                st = T()
                for i in range(self.num_gpus):
                    delta_path = get_slice_path(window, self.window_to_seed(window), f'delta_{i}')
                    delta = torch.load(delta_path)
                    # Apply the delta to the main model.
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            param.data += delta[name].to(param.device)
                logger.info(f"{P(window, T() - st)}: Aggregated and applied deltas from all slices.")

                # Upload the state for the next window.
                st = T()
                await save_slice_for_window(
                    model_state=self.model.state_dict(),
                    window=window + 1,
                    seed=self.window_to_seed(window + 1),
                    compression=self.hparams.compression,
                    key='state'
                )
                logger.info(f"{P(window, T() - st)}: Uploaded model state for window {window + 1}.")

                # Clean file history.
                st = T()
                await delete_files_before_window(window_max=window - self.hparams.max_history, key='state')
                for i in range(self.num_gpus):
                    await delete_files_before_window(window_max=window - self.hparams.max_history, key=f'delta_{i}')
                    await delete_files_from_bucket_before_window(bucket=self.config.bucket, window_max=window - self.hparams.max_history, key=f'delta_{i}')
                await delete_files_from_bucket_before_window(bucket=self.config.bucket, window_max=window - self.hparams.max_history, key='state')
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
                        "loss": avg_loss,
                        "tokens_per_step": tokens_per_step,
                        "tokens_per_second": tokens_per_second,
                        "sample_rate": self.sample_rate,
                        "utilization": train_duration / (end_step - start_step),
                        "learning_rate": self.scheduler.get_last_lr()[0]
                    })

            except KeyboardInterrupt:
                logger.info("Training interrupted by user. Stopping the run.")
                self.stop_event.set()
                await self.update_task
                sys.exit(0)

            except Exception as e:
                logger.exception(f"Exception during training loop: {e}")
                continue

    def block_to_window(self, block: int) -> int:
        """Convert a block number to a window number.

        Args:
            block (int): The block number.

        Returns:
            int: The corresponding window number.
        """
        return int(block / self.hparams.window_length)

    def window_to_seed(self, window: int) -> int:
        """Generate a seed based on the window number.

        Args:
            window (int): The window number.

        Returns:
            int: The seed value.
        """
        # Original implementation for single GPU.
        return int(self.subtensor.get_block_hash(window * self.hparams.window_length), 16)

    def window_to_seeds(self, window: int, num_slices: int) -> List[int]:
        """Generate seeds for multiple slices based on the window number.

        If `num_slices` is 1, it returns a list with a single seed using the original `window_to_seed` method.

        Args:
            window (int): The window number.
            num_slices (int): Number of slices (GPUs).

        Returns:
            List[int]: A list of seed values for each slice.
        """
        if num_slices == 1:
            # Single GPU scenario: use the original seed.
            return [self.window_to_seed(window)]
        else:
            # Multi-GPU scenario: generate different seeds for each slice.
            base_seed = int(self.subtensor.get_block_hash(window * self.hparams.window_length), 16)
            seeds = []
            for i in range(num_slices):
                # Combine base seed with slice index and miner UID to generate unique seeds.
                seed = hash((base_seed, i, self.uid)) & 0xffffffff
                seeds.append(seed)
            return seeds

    def block_listener(self, loop: asyncio.AbstractEventLoop):
        """Listener thread that updates the current block and window when a new block is announced.

        Args:
            loop (asyncio.AbstractEventLoop): The event loop for scheduling tasks.
        """
        def handler(event: Dict[str, Any], _u, _s):
            self.current_block = int(event['header']['number'])
            loop.call_soon_threadsafe(self.new_block_event.set)
            if self.block_to_window(self.current_block) != self.current_window:
                self.current_window = self.block_to_window(self.current_block)
                self.window_seeds[self.current_window] = self.window_to_seed(self.current_window)
                self.window_duration = T() - self.window_time if hasattr(self, 'window_time') else 0
                self.window_time = T()
                loop.call_soon_threadsafe(self.new_window_event.set)
                logger.info(f"{P(self.current_window, self.window_duration)} New Window.")

        # Subscribe to block headers with retry logic.
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)
                break
            except Exception as e:
                logger.error(f"Failed to subscribe to block headers: {e}. Retrying in 1 second...")
                time.sleep(1)

    async def update_weights(self):
        """Update weights periodically based on the latest metagraph."""
        while not self.stop_event.is_set():
            try:
                # Update metagraph.
                self.metagraph.sync()
                # Update buckets list.
                self.buckets = [self.config.bucket]
                if self.config.remote:
                    self.buckets.extend(self.metagraph.buckets)
                await asyncio.sleep(self.hparams.metagraph_update_interval)
            except Exception as e:
                logger.exception(f"Exception in update_weights: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(Miner().run())