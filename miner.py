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

# TODO: formalised : Add attribution

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
from torch.distributed.elastic.multiprocessing.errors import record
import sys
import traceback


# Import local files.
from boltz.common import *
from hparams import load_hparams
from boltz.datasets.dataset import *
from boltz.utils import *
from boltz.models import model_name_to_cls, model_name_to_tokenizer, models_config
from boltz.config_manager import JobConfig
from boltz.datasets.tokenizer import build_tokenizer
from boltz.parallelisms import (
    models_parallelize_fns,
    ParallelDims,
)
from boltz.checkpoint import CheckpointManager, TrainState
from boltz.metrics import build_gpu_memory_monitor, build_metric_logger
from boltz.optimizer import build_lr_schedulers, build_optimizers

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Miner:
    
    def __init__(self):
        # Init config.
        self.config = JobConfig()
        logger.info('\n' + '-' * 40 + ' Config ' + '-' * 40)
        logger.info(f"Starting job: {self.config.job.description}")

        # used for colorful printing
        color = Color if self.config.metrics.enable_color_printing else NoColor
        logger.info(self.config)

        # Init bittensor objects.
        self.wallet = bt.wallet(config=self.config.config)
        self.subtensor = bt.subtensor(config=self.config.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.bittensor.netuid)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise ValueError(f'Wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}')
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info('\n' + '-' * 40 + ' Objects ' + '-' * 40)
        logger.info(f'\nWallet: {self.wallet}\nSubtensor: {self.subtensor}\nMetagraph: {self.metagraph}\nUID: {self.uid}')

        # Init bucket.
        try:
            if self.config.bittensor.bucket != self.subtensor.get_commitment(self.config.bittensor.netuid, self.uid):
                raise ValueError('')
        except:
            self.subtensor.commit(self.wallet, self.config.bittensor.netuid, self.config.bittensor.bucket)
        logger.info('Bucket:' + self.config.bittensor.bucket)

        # Init Wandb.
        if self.config.bittensor.use_wandb:
            # Delete all runs with my name and create a new one.
            try:
                for run in wandb.Api().runs(path=self.config.bittensor.project):
                    if run.name == f'M{self.uid}':
                        logger.info(f'Deleting old run: {run}'); run.delete()
            except: pass
            wandb.init(project=self.config.bittensor.project, resume='allow', name=f'M{self.uid}', config=self.config)

        # Init model.
        logger.info('\n' + '-' * 40 + ' Hparams ' + '-' * 40)
        self.hparams = load_hparams()
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        # take control of garbage collection to avoid stragglers
        self.gc_handler = GarbageCollection(gc_freq=self.config.training.gc_freq)

        set_determinism(self.config.training.seed)
        if self.config.training.seed is None:
            logger.info("Deterministic training off")
        else:
            logger.info(
                f"Deterministic training on. Using seed: {self.config.training.seed}"
            )
        # init distributed
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.parallel_dims = ParallelDims(
            dp_shard=self.config.training.data_parallel_shard_degree,
            dp_replicate=self.config.training.data_parallel_replicate_degree,
            tp=self.config.training.tensor_parallel_degree,
            pp=self.config.experimental.pipeline_parallel_degree,
            world_size=self.world_size,
            enable_loss_parallel=self.config.training.enable_loss_parallel,
        )
        self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        torch.cuda.set_device(self.device)
        init_distributed(self.config)
        # initialize GPU memory monitor and get peak flops for MFU calculation
        self.gpu_memory_monitor = build_gpu_memory_monitor()
        self.gpu_peak_flops = get_peak_flops(self.gpu_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {self.gpu_peak_flops:.3e}")

        self.metric_logger = build_metric_logger(self.config, self.parallel_dims)

        # build meshes
        world_mesh = self.parallel_dims.build_mesh(device_type="cuda")
        if self.parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            self.dp_degree, self.dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()

        model_name = self.config.model.name

        # build tokenizer
        tokenizer_type = model_name_to_tokenizer[model_name]
        self.tokenizer = build_tokenizer(tokenizer_type, self.config.model.tokenizer_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_id
        # build model (using meta init)
        model_cls = model_name_to_cls[model_name]
        model_config = models_config[model_name][self.config.model.flavor]
        model_config.norm_type = self.config.model.norm_type
        model_config.vocab_size = self.tokenizer.n_words
        model_config.max_seq_len = self.config.training.seq_len

        logger.info(f"Building {model_name} {self.config.model.flavor} with {model_config}")
        with torch.device("meta"):
            self.model = model_cls.from_model_args(model_config)

        # # a no-op hander if float8 is not enabled
        # float8_handler = Float8Handler(config, parallel_dims)
        # # swap to Float8Linear based on float8 configs
        # float8_handler.convert_to_float8_training(model)

        # log model size
        self.model_param_count = get_num_params(self.model)
        self.num_flop_per_token = get_num_flop_per_token(
            get_num_params(self.model, exclude_embedding=True),
            model_config,
            self.config.training.seq_len,
        )
        logger.info(
            f"{color.blue}Model {self.config.model.name} {self.config.model.flavor} "
            f"{color.red}size: {self.model_param_count:,} total parameters{color.reset}"
        )

        # loss function to be shared by Pipeline Parallel and SPMD training
        def loss_fn(pred, labels):
            return torch.nn.functional.cross_entropy(
                pred.flatten(0, 1), labels.flatten(0, 1)
            )

        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel (Ingoring TP/ SP for now)
        models_parallelize_fns[model_name](self.model, world_mesh, self.parallel_dims, self.config)

        # move sharded model to CPU/GPU and initialize weights via DTensor
        init_device = "cpu" if self.config.checkpoint.create_seed_checkpoint else "cuda"
        self.model.to_empty(device=init_device)
        self.model.init_weights()
        self.model.train()

        self.model_parts = [self.model]
        gpu_mem_stats = self.gpu_memory_monitor.get_peak_stats()
        logger.info(
            f"GPU memory usage for model: "
            f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
            f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = build_optimizers(self.model_parts, self.config)
        self.lr_schedulers = build_lr_schedulers(self.optimizers.optimizers, self.config)

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
        self.last_full_steps = self.hparams.desired_batch_size // self.config.bittensor.actual_batch_size    
        print ( self.hparams )
    
    @record    
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
    @record
    async def run(self):
        # Main loop.
        self.loop = asyncio.get_running_loop()
        self.update_task = asyncio.create_task(self.update())
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()
        
        # Optionally sync the model state by pulling model states from the history.
        # if self.config.bittensor.sync_state:
        #     history_windows = [ self.current_window - i for i in range (self.hparams.max_history) ]
        #     state_slices = await download_slices_for_buckets_and_windows(
        #         buckets = self.buckets,
        #         windows = history_windows,
        #         key = 'state'
        #     )
        #     for window in tqdm(history_windows, desc="Syncing state"):
        #         await apply_slices_to_model( 
        #             model = self.model, 
        #             window = window,
        #             seed = window,
        #             compression = self.hparams.compression,
        #             key = 'state'
        #         )
            # torch.cuda.empty_cache()
            
        # Main training loop.
        while True:
            try:      
                # Start the window step.     
                logger.info('[bold]' + '\n' + '-' * 40 + f' Step: {self.global_step} ' + '-' * 40)
                self.global_step += 1
                start_step = T()
                window = self.current_window
                
                # Run for non-baseline miners.
                if not self.config.bittensor.baseline:
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
                    
                    start_step = T()
                    window = self.current_window
                    current_window = window  # Set current_window at the start

                    # Fetch pages for the current window
                    st = T()
                    pages = await DatasetLoader.next_pages(
                        offset=window,
                        n_pages=self.hparams.validator_window_eval_size,
                        seed=self.uid if not self.config.bittensor.random else random.randint(0, 1000),
                        num_rows_per_page=100  # Adjust as needed
                    )

                    # Create the dataset asynchronously
                    dataset = await DatasetLoader.create(
                        batch_size=self.config.bittensor.actual_batch_size,
                        sequence_length=self.hparams.sequence_length,
                        pages_info=pages,
                        tokenizer=self.tokenizer,
                        pack_samples=False
                    )
                    logger.info(f"{P(window, T() - st)}: Downloaded training pages: {pages}")

                    # Initialize data iterator directly from the dataset
                    data_iterator = iter(dataset)

                    # Training loop variables
                    train_state = TrainState()
                    losses_since_last_log = []
                    ntokens_since_last_log = 0
                    data_loading_times = []
                    time_last_log = time.perf_counter()
                    total_loss = 0.0
                    processed_steps = 0
                    total_steps = 0
                    exhausted_window = False
                    sample_rate = self.config.training.sample_rate

                    # Training loop
                    while not exhausted_window:
                        train_state.step += 1
                        self.gc_handler.run(train_state.step)

                        # Check if the training window has changed
                        window = self.current_window
                        if window != current_window and not self.config.bittensor.baseline:
                            exhausted_window = True
                            continue

                        # Get batch
                        data_load_start = time.perf_counter()
                        try:
                            batch = next(data_iterator)
                        except StopIteration:
                            # Data exhausted
                            logger.info("Dataset exhausted.")
                            exhausted_window = True
                            continue
                        data_loading_times.append(time.perf_counter() - data_load_start)

                        # Apply sample rate logic
                        if random.random() < sample_rate:
                            total_steps += 1
                            processed_steps += 1

                            # Prepare input_ids and labels
                            input_ids = torch.tensor(batch, dtype=torch.long).to(self.device)
                            labels = input_ids.clone()
                            labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                            ntokens_since_last_log += labels.numel()
                        else:
                            # Skip this batch
                            continue

                        self.optimizers.zero_grad()
                        self.model.train()
                        step_loss = 0 
                        for input_ids, labels in dataset:
                            input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
                            labels = torch.tensor(labels, dtype=torch.long).to(self.device)

                            # Handle padding tokens in labels if necessary
                            labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)

                            self.optimizers.zero_grad()
                            pred = self.model(input_ids)
                            loss = self.loss_fn(pred.view(-1, pred.size(-1)), labels.view(-1))
                            loss.backward()

                            del pred

                            step_loss += loss.item()
                            for m in self.model_parts:
                                torch.nn.utils.clip_grad_norm_(
                                    m.parameters(), self.config.training.max_norm, foreach=True
                                )

                        self.optimizers.step()
                        self.lr_scheduler.step()
                        self.optimizers.zero_grad()

                        # Update training state and metrics
                        losses_since_last_log.append(loss.item())
                        train_state.log_steps.append(train_state.step)
                        ntokens_since_last_log += labels.numel()

                        # Logging and metrics
                        if train_state.step % self.config.metrics.log_freq == 0:
                            # Compute average and max loss
                            avg_loss = sum(losses_since_last_log) / len(losses_since_last_log)
                            max_loss = max(losses_since_last_log)

                            # Compute global average and max loss
                            global_avg_loss = avg_loss
                            global_max_loss = max_loss

                            # Compute time delta and performance metrics
                            time_delta = time.perf_counter() - time_last_log
                            wps = ntokens_since_last_log / (time_delta * self.parallel_dims.model_parallel_size)
                            mfu = 100 * self.num_flop_per_token * wps / self.gpu_peak_flops

                            # Compute data loading time statistics
                            time_end_to_end = time_delta / self.config.metrics.log_freq
                            time_data_loading = sum(data_loading_times) / len(data_loading_times)
                            time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                            # Get GPU memory stats
                            gpu_mem_stats = self.gpu_memory_monitor.get_peak_stats()

                            # Define tokens_per_step
                            tokens_per_step = ntokens_since_last_log / self.config.metrics.log_freq

                            # Define step_loss as the average loss since last log
                            step_loss = avg_loss

                            # Define train_duration
                            train_duration = time_delta

                            # Update end_step and start_step
                            end_step = train_state.step
                            start_step = end_step - self.config.metrics.log_freq + 1

                            # Get current learning rate
                            current_lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

                            # Prepare metrics dictionary
                            metrics = {
                                "loss_metrics/global_avg_loss": global_avg_loss,
                                "loss_metrics/global_max_loss": global_max_loss,
                                "wps": wps,
                                "mfu(%)": mfu,
                                "time_metrics/end_to_end(s)": time_end_to_end,
                                "time_metrics/data_loading(s)": time_data_loading,
                                "time_metrics/data_loading(%)": time_data_loading_pct,
                                "memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                                "memory/max_active(%)": gpu_mem_stats.max_active_pct,
                                "memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                                "memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                                "memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                                "memory/num_ooms": gpu_mem_stats.num_ooms,
                                "train/processed_steps": processed_steps,
                                "train/total_steps": total_steps,
                                "train/sample_rate": sample_rate,
                                "loss": step_loss,
                                "tokens_per_step": tokens_per_step,
                                "tokens_per_second": wps,
                                "sample_rate": sample_rate,
                                "utilization": train_duration / (end_step - start_step + 1),
                                "learning_rate": current_lr,
                            }

                            # Log metrics using metric_logger and wandb
                            self.metric_logger.log(metrics, step=train_state.step)
                            if self.config.bittensor.use_wandb:
                                wandb.log(metrics)

                            # Log to console
                            logger.info(
                                f"Step: {train_state.step:2}  "
                                f"Loss: {global_avg_loss:7.4f}  "
                                f"Memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                                f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                                f"WPS: {round(wps):,}  "
                                f"MFU: {mfu:.2f}%"
                            )

                            # Reset counters
                            losses_since_last_log.clear()
                            ntokens_since_last_log = 0
                            data_loading_times.clear()
                            time_last_log = time.perf_counter()
                            self.gpu_memory_monitor.reset_peak_stats()

                        # Adjust sample rate dynamically
                        if exhausted_window:
                            sample_rate = max(0.0001, sample_rate * 0.95)
                        else:
                            sample_rate = min(1.0, sample_rate * 1.05)

                        # Update wandb config
                        if self.config.bittensor.use_wandb:
                            wandb.config.update({'sample_rate': sample_rate}, allow_val_change=True)
                    # Save checkpoint
                    #  @sd: checkpointing not implemented yet
                    # checkpoint.save(
                    #     train_state.step,
                    #     force=(train_state.step == self.config.training.steps)
                    # )

                # Run for non-baseline nodes.
                if not self.config.bittensor.baseline:
                    # Upload the delta for the previous window.
                    st = T()
                    # await upload_slice_for_window(
                    #     bucket = self.config.bittensor.bucket, 
                    #     model = self.model, 
                    #     window = window,
                    #     seed = window,
                    #     wallet = self.wallet, 
                    #     compression = self.hparams.compression,
                    #     key = 'delta'
                    # )                
                    logger.info(f"{P(window, T() - st)}: Uploaded the delta.")
                    
                    # Apply the delta from the previous window.
                    st = T()
                    # await apply_slices_to_model(
                    #     model = self.model, 
                    #     window = window - 1,
                    #     seed = window - 1,
                    #     compression = self.hparams.compression,
                    #     key = 'delta'
                    # )         
                    logger.info(f"{P(window, T() - st)}: Applied window delta.")
                
                    # Upload the state for the current window.
                    st = T()
                    # await upload_slice_for_window(
                    #     bucket = self.config.bittensor.bucket, 
                    #     model = self.model, 
                    #     window = window + 1,
                    #     seed = window + 1, 
                    #     wallet = self.wallet, 
                    #     compression = self.hparams.compression,
                    #     key = 'state',
                    # )
                    # logger.info(f"{P(window, T() - st)}: Uploaded the state.")
                    
                    # Clean file history.
                    st = T()
                    await delete_files_before_window( window_max = window - self.hparams.max_history, key = 'state')
                    await delete_files_before_window( window_max = window - self.hparams.max_history, key = 'delta')
                    await delete_files_from_bucket_before_window( bucket = self.config.bittensor.bucket, window_max = window - self.hparams.max_history, key = 'state' )
                    await delete_files_from_bucket_before_window( bucket = self.config.bittensor.bucket, window_max = window - self.hparams.max_history, key = 'delta' )
                    logger.info(f"{P(window, T() - st)}: Cleaned file history.")
                    
                    # Wait until we are on a new window.
                    end_step = T()
                    while self.current_window == window:
                        await asyncio.sleep(0.1)
                    window_time_delta = self.window_time - end_step
                    window_delta_str = f"[red]{window_time_delta:.2f}[/red]" if window_time_delta < 0 else f"[green]+{window_time_delta:.2f}[/green]"
                    logger.info(f"{P(window, end_step - start_step)}[{window_delta_str}]: Finished step.")
                    if self.config.bittensor.use_wandb:
                        wandb.log({
                            f"loss": step_loss,
                            f"tokens_per_step": tokens_per_step,
                            f"tokens_per_second": wps,
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
    @record
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
                bt.subtensor(config=self.config.config).substrate.subscribe_block_headers(handler); break
            except Exception as e:
                 # Wait for 5 seconds before retrying
                logger.error(f"Failed to subscribe to block headers: {e}.\nRetrying in 1 seconds...")
                time.sleep(1) 
            
if __name__ == "__main__":
    try:
        asyncio.run(Miner().run())
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)