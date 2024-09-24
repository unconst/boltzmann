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

from queue import Queue
import io
import os
import uuid
import time
import wandb
import boto3
import torch
import tempfile
import argparse
import traceback
import numpy as np
import threading
from tqdm import tqdm
import bittensor as bt
import concurrent.futures
import torch.optim as optim
from typing import List, Optional, Tuple, Dict, Any, Set
from dotenv import dotenv_values
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import CosineAnnealingLR
import threading
from queue import Queue

from hparams import load_hparams
from dataset import SubsetFineWebEdu2Loader

# Enable cuDNN benchmark for optimized performance
torch.backends.cudnn.benchmark = True

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Instantiate the AWS S3 client.
env_config = {**dotenv_values(".env"), **os.environ}  # Load environment variables.
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')  # AWS access key ID.
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')  # AWS secret access key.
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',  # AWS region.
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Initialize a global ThreadPoolExecutor for mask downloads.
mask_download_executor = concurrent.futures.ThreadPoolExecutor(max_workers=10) 

def main(config):
    """
    Main function that runs the miner script.

    Args:
        config: Configuration object with all the necessary parameters.

    Returns:
        None
    """

    # Print the configuration settings.
    print('\n', '-' * 40, 'Config', '-' * 40)
    print(config)
    
    # Initialize Bittensor objects.
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    print('\n', '-' * 40, 'Objects', '-' * 40)
    print(f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}')
    
    # Initialize bucket commitment.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception:
        # If not committed or mismatch, commit the bucket to the chain.
        subtensor.commit(wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket)

    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        run = wandb.init(project='cont', resume='allow', name=f'M{my_uid}', config=config)
        
    # Initialize training state.
    hparams = load_hparams()
    model = None
    upload_history = []
    last_mask_sync = 0
    last_master_sync = 0

    # Initialize buckets and related variables.
    buckets: List[str] = []
    last_bucket_update: float = 0
    bucket_update_interval: float = 300  # Update every 5 minutes.
    buckets_lock = threading.Lock()
    uid_to_bucket: Dict[int, Optional[str]] = {} 


    def update_buckets():
        """
        Fetch and store bucket commitments for all UIDs.

        This function updates the `uid_to_bucket` dictionary with the latest commitments
        from the chain for all UIDs in the metagraph.

        It's optimized for synchronous execution without threading.

        Example:
            update_buckets()
        """
        nonlocal uid_to_bucket, last_bucket_update, subtensor, config, metagraph

        print("Updating commitments from buckets...")

        # Initialize the mapping from UID to bucket
        new_uid_to_bucket: Dict[int, Optional[str]] = {}

        # Fetch commitments synchronously
        for idx in range(len(metagraph.uids)):
            uid = metagraph.uids[idx]
            try:
                # Fetch the commitment (bucket name) for the current UID
                bucket_commitment = subtensor.get_commitment(config.netuid, uid)
                new_uid_to_bucket[uid] = bucket_commitment
            except Exception as e:
                # If an error occurs, set the commitment to None and log the error
                print(f"Error fetching commitment for UID {uid}: {e}")
                new_uid_to_bucket[uid] = None

        # Acquire the lock before updating the shared uid_to_bucket mapping
        with buckets_lock:
            uid_to_bucket = new_uid_to_bucket

        # Update the timestamp of the last bucket update
        last_bucket_update = time.time()
        print("Bucket commitments updated successfully.")

    def get_available_masks(uid_to_bucket: Dict[int, str], all_sync_blocks: List[int]) -> Dict[int, Set[int]]:
        """
        Get a mapping of blocks to UIDs who have masks available.

        Args:
            uid_to_bucket (Dict[int, str]): Mapping of UIDs to their corresponding buckets.
            all_sync_blocks (List[int]): List of blocks to check.

        Returns:
            Dict[int, Set[int]]: Mapping of block numbers to sets of UIDs who have masks available for those blocks.
        """
        block_to_uids: Dict[int, Set[int]] = {}

        # Iterate over all UIDs and their buckets
        for uid, bucket in uid_to_bucket.items():
            if not bucket:
                continue

            hotkey = metagraph.hotkeys[uid]
            prefix = f"mask-{hotkey}-"

            try:
                paginator = CLIENT.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        # Extract the block number from the filename
                        parts = key.split('-')
                        if len(parts) >= 3:
                            block_str = parts[-1].split('.')[0]  # Get block number
                            try:
                                block_num = int(block_str)
                                if block_num in all_sync_blocks:
                                    if block_num not in block_to_uids:
                                        block_to_uids[block_num] = set()
                                    block_to_uids[block_num].add(uid)
                            except ValueError:
                                continue
            except Exception as e:
                print(f"Error listing masks for UID {uid} in bucket {bucket}: {e}")

        return block_to_uids

    while True:
        try:
            # Sync the current chain state and hparams periodically.
            print('Loading chain state...')
            chain_state_last_updated = time.time()
            chain_state_update_interval = 60  # Update every 60 seconds.
            current_time = time.time()
            if current_time - chain_state_last_updated >= chain_state_update_interval:
                print('Loading chain state...')
                hparams = load_hparams()
                subtensor = bt.subtensor(config=config)
                metagraph = subtensor.metagraph(netuid=config.netuid)
                chain_state_last_updated = current_time
                print(f'Chain state loaded.')

            # Update buckets if needed.
            if time.time() - last_bucket_update > bucket_update_interval:
                print('Updating buckets...')
                bucket_update_start_time = time.time()
                # Update buckets synchronously
                update_buckets()
                last_bucket_update = time.time()
                print(f'Bucket update completed in {time.time() - bucket_update_start_time:.2f} seconds')

            # Synchronize the full model state every hparams.epoch_length.
            if model is None or subtensor.block - last_master_sync > hparams.epoch_length:
                print('Synchronizing model with master...')
                try:
                    master_uid = int(metagraph.S.argmax())
                    master_bucket = subtensor.get_commitment(config.netuid, master_uid)
                    master_hotkey = metagraph.hotkeys[master_uid]
                    master_filename = f'master-{master_hotkey}.pt'
                    unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                    CLIENT.download_file(master_bucket, master_filename, unique_temp_file)
                    master_state_dict = torch.load(unique_temp_file, map_location='cpu', weights_only=True)
                    model = LlamaForCausalLM(config=hparams.model_config)
                    model.load_state_dict(master_state_dict)

                    model.to(config.device)
                    model.train()
                    optimizer = optim.AdamW(
                        model.parameters(),
                        lr = config.learning_rate,  # Peak learning rate
                        betas = ( config.optimizer_beta1, config.optimizer_beta2 ), # B1 and B2
                        weight_decay = config.optimizer_weight_decay,  # Weight decay
                        foreach = True,  # more memory usage, but faster
                    )
                    scheduler = CosineAnnealingLR( optimizer, T_max = hparams.epoch_length, eta_min=4e-5, last_epoch=-1 )
                    last_master_sync = subtensor.block 
                    last_mask_sync = last_master_sync
                except Exception as e:
                    print(f'No master model available: {e}. Waiting...')
                    time.sleep(12)
                    continue

            # Prepare for asynchronous mask downloading.
            print('Preparing for mask downloads...')
            mask_download_start_time = time.time()
            block = subtensor.block  # Get the current block number
            # Define all_sync_blocks using last_mask_sync and the current block
            all_sync_blocks = list(range(last_mask_sync + 1, block + 1))  # Blocks to sync
            last_mask_sync = subtensor.block
            # blocks_with_masks, mask_filename_to_bucket = get_blocks_with_masks_and_mapping(buckets, all_sync_blocks)

            # Get the mapping of blocks to UIDs with available masks
            # TODO: time
            print('Listing available masks...')
            block_to_uids = get_available_masks(uid_to_bucket, all_sync_blocks)
            print(f'Blocks with available masks: {list(block_to_uids.keys())}')

            mask_downloads: Dict[int, List[concurrent.futures.Future]] = {}
            # Initialize a dictionary to hold metadata about each mask file
            mask_metadata = {}

            # Only process blocks that have masks
            if not block_to_uids:
                print('No masks available for the blocks in this range. Skipping mask downloading and processing.')
            else:
                # Aggregate the blocks into a sorted list
                blocks_with_masks = sorted(block_to_uids.keys())
                print(f'Starting asynchronous mask downloads for blocks: {blocks_with_masks}')
                
                # Start asynchronous downloads using the block_to_uids mapping
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    for blk in blocks_with_masks:
                        futures = []
                        for uid in block_to_uids[blk]:
                            with buckets_lock:
                                bucket = uid_to_bucket.get(uid)
                            if not bucket:
                                continue

                            hotkey = metagraph.hotkeys[uid]
                            filename = f"mask-{hotkey}-{blk}.pt"

                            future = executor.submit(download_file, bucket, filename)
                            future.uid = uid
                            future.hotkey = hotkey
                            future.blk = blk
                            futures.append(future)
                        mask_downloads[blk] = futures
                print('All mask downloads have been initiated.')

            print(f'Prepared mask downloads in {time.time() - mask_download_start_time:.2f} seconds')

            # Start fetching pages asynchronously for the next iteration.
            print('Fetching pages...')
            fetch_start_time = time.time()
            n_pages = max(1, int(config.desired_batch_size * 0.01))
            result_holder = {}
            fetch_thread = threading.Thread(
                target=fetch_pages_async,
                args=(subtensor, n_pages, my_uid, result_holder)
            )
            fetch_thread.start()

            # Proceed with training while masks are downloading and pages are fetching.
            print('Starting training...')
            torch.cuda.empty_cache() # Empty cache going into the training step.
            training_start_time = time.time()

            total_loss = 0.0
            total_steps = config.desired_batch_size // config.actual_batch_size

            # Initialize dataset and data loader.
            fetch_thread.join()
            if 'error' in result_holder:
                raise result_holder['error']
            else:
                pages = result_holder['pages']

            dataset = SubsetFineWebEdu2Loader(
                batch_size=config.actual_batch_size,
                sequence_length=hparams.sequence_length,
                pages_info=pages,
                tokenizer=hparams.tokenizer
            )
            data_loader = PrefetchingDataLoader(dataset)

            progress_bar = tqdm(total=total_steps, desc="Training:")
            for idx, batch in enumerate(data_loader):
                optimizer.zero_grad()
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                with torch.amp.autocast( device_type = model.device.type, dtype = torch.bfloat16 ):  # Enable autocasting for mixed precision
                    outputs = model(input_ids = input_ids, labels=labels)
                total_loss += outputs.loss.item()
                outputs.loss.backward()
                progress_bar.update(1)  # Update the progress bar
                if idx >= total_steps - 1:
                    break
            progress_bar.close()

            # Optimizer step.
            try:

                # grad norm clipping
                if config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()  # Update the learning rate.
                optimizer.zero_grad()
            except AssertionError as e:
                print(f"An error occurred during the optimizer step: {e}")

            # Calculate and print average loss.
            average_loss = total_loss / total_steps
            training_time = time.time() - training_start_time
            steps_per_second = total_steps / training_time
            tokens_per_second = hparams.sequence_length * config.actual_batch_size * total_steps / training_time
            print(f'Training completed in {training_time:.2f} seconds')
            print(f'Average Loss: {average_loss:.6f}')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6e}')
            print(f'Steps per second: {steps_per_second:.2f}')
            print(f'Tokens per second: {tokens_per_second:.2f}')
            if config.use_wandb:
                wandb.log({
                    "step_loss": average_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    f"incentive_{my_uid}": float(metagraph.I[my_uid]),
                    "training_time": training_time,
                    "steps_per_second": steps_per_second,
                    "tokens_per_second": tokens_per_second
                })

            # Collect and process the downloaded masks after training.
            mask_processing_start_time = time.time()
            for blk, futures in mask_downloads.items():
                print(f'Processing downloaded masks for block: {blk}')
                temp_files = []
                n_downloaded = 0
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        uid = future.uid
                        hotkey = future.hotkey
                        temp_files.append(result)
                        mask_metadata[result] = (uid, hotkey)
                        n_downloaded += 1
                print(f'Downloaded {n_downloaded} masks for block {blk}')

                if n_downloaded == 0:
                    continue

                # Process masks as before.
                process_start_time = time.time()
                process_downloaded_masks(model, temp_files, blk, hparams, config,mask_metadata)
                processing_time = time.time() - process_start_time
                print(f'Processed masks for block {blk} in {processing_time:.2f} seconds')



                # Clean up temporary files.
                for file in temp_files:
                    os.remove(file)
            # sd: do we need this ?
            mask_processing_time = time.time() - mask_processing_start_time
            print(f'Processed downloaded masks in {mask_processing_time:.2f} seconds')

            # Create and upload the mask of the model asynchronously.
            print('Creating and uploading mask...')
            mask_creation_start_time = time.time()
            next_upload_block = subtensor.block
            model_state_dict = create_upload_mask(model, next_upload_block, hparams)
            upload_thread = threading.Thread(
                target=upload_mask_async,
                args=(model_state_dict, wallet.hotkey.ss58_address, next_upload_block, config, CLIENT, upload_history)
            )
            upload_thread.start()                 
            # Ensure the upload is finished before next iteration.
            mask_creation_time = time.time() - mask_creation_start_time
            print(f'Created and uploaded mask in {mask_creation_time:.2f} seconds')

            # Clean up GPU memory.
            torch.cuda.empty_cache()
            print('Iteration completed.\n')

        # Handle keyboard interrupts for graceful shutdown.
        except (KeyboardInterrupt, SystemExit):
            print("Training interrupted. Exiting gracefully.")
            break

        # Handle any other exceptions.
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(5)
            continue

    # Shutdown the executor when exiting the main loop
    mask_download_executor.shutdown(wait=False)

def download_file(bucket: str, filename: str) -> str:
    """
    Download a file from S3 if it exists.

    Args:
        bucket (str): The S3 bucket name.
        filename (str): The filename in the S3 bucket.

    Returns:
        str: The local path to the downloaded file or None if file does not exist.
    """
    try:
        # Check if the object exists
        CLIENT.head_object(Bucket=bucket, Key=filename)

        # If it exists, proceed to download
        unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
        CLIENT.download_file(bucket, filename, unique_temp_file)
        return unique_temp_file
    except CLIENT.exceptions.ClientError as e:
        # # If a 404 error is returned, then the object does not exist
        # if e.response['Error']['Code'] == '404':
        #     print(f"File {filename} does not exist in bucket {bucket}.")
        # else:
        #     print(f"Error checking existence of {filename} in bucket {bucket}: {e}")
        return None
    except Exception as e:
        print(f"Download error for {filename} from bucket {bucket}: {e}")
        return None

def get_blocks_with_masks_and_mapping(buckets: List[str], starting_block: int, ending_block: int) -> Tuple[Set[int], Dict[str, str]]:
    """
    Get a set of blocks for which masks exist and a mapping of mask filenames to buckets.

    Args:
        buckets (List[str]): List of buckets to check.
        starting_block (int): Starting block number.
        ending_block (int): Ending block number.

    Returns:
        Tuple[Set[int], Dict[str, str]]: 
            - Set of blocks that have at least one mask available.
            - Dictionary mapping mask filenames to their respective buckets.
    """
    blocks_with_masks = set()
    mask_filename_to_bucket = {}
    for bucket in set(buckets):
        if bucket is None:
            continue
        try:
            paginator = CLIENT.get_paginator('list_objects_v2')
            prefix = 'mask-'
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    # Extract the block number from the filename
                    parts = key.split('-')
                    if len(parts) >= 3:
                        block_str = parts[-1].split('.')[0]  # Get block number
                        try:
                            block_num = int(block_str)
                            if starting_block <= block_num <= ending_block:
                                blocks_with_masks.add(block_num)
                                mask_filename_to_bucket[key] = bucket
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Error listing objects in bucket {bucket}: {e}")
    return blocks_with_masks, mask_filename_to_bucket



def get_blocks_with_masks(buckets: List[str], starting_block: int, ending_block: int) -> Set[int]:
    """
    Get a set of blocks for which masks exist.

    Args:
        buckets (List[str]): List of buckets to check.
        starting_block (int): Starting block number.
        ending_block (int): Ending block number.

    Returns:
        Set[int]: Set of blocks that have at least one mask available.
    """
    blocks_with_masks = set()
    for bucket in set(buckets):
        if bucket is None:
            continue
        try:
            paginator = CLIENT.get_paginator('list_objects_v2')
            prefix = 'mask-'
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    # Extract the block number from the filename
                    # Assuming filename format: 'mask-{hotkey}-{block}.pt'
                    parts = key.split('-')
                    if len(parts) >= 3:
                        block_str = parts[-1].split('.')[0]  # Get block number
                        try:
                            block_num = int(block_str)
                            if starting_block <= block_num <= ending_block:
                                blocks_with_masks.add(block_num)
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Error listing objects in bucket {bucket}: {e}")
    return blocks_with_masks

def fetch_pages_async(subtensor: Any, n_pages: int, my_uid: int, result_holder: Dict[str, Any]):
    """
    Fetch pages asynchronously for data loading.

    Args:
        subtensor (Any): The subtensor object.
        n_pages (int): Number of pages to fetch.
        my_uid (int): The UID of the miner.
        result_holder (dict): Dictionary to hold the result or errors.

    Returns:
        None
    """
    try:
        pages = SubsetFineWebEdu2Loader.next_pages(
            offset=subtensor.block + n_pages,
            n_pages=n_pages,
            seed=my_uid
        )
        result_holder['pages'] = pages
    except Exception as e:
        result_holder['error'] = e

class PrefetchingDataLoader:
    """
    DataLoader that pre-fetches data batches in a background thread.

    Args:
        data_loader (Iterable): The data loader to wrap.
        prefetch_size (int): Number of batches to prefetch.

    Example:
        data_loader = PrefetchingDataLoader(dataset)
        for batch in data_loader:
            # Training code here.
    """
    def __init__(self, data_loader, prefetch_size=2):
        self.data_loader = data_loader
        self.prefetch_size = prefetch_size
        self.queue = Queue(maxsize=prefetch_size)
        self.iterator = iter(data_loader)
        self.stop_signal = object()
        self.thread = threading.Thread(target=self._prefetch)
        self.thread.daemon = True
        self.thread.start()

    def _prefetch(self):
        """Background thread function to prefetch data."""
        try:
            for batch in self.iterator:
                self.queue.put(batch)
            self.queue.put(self.stop_signal)
        except Exception as e:
            self.queue.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.queue.get()
        if batch is self.stop_signal:
            raise StopIteration
        elif isinstance(batch, Exception):
            raise batch
        else:
            return batch

def process_downloaded_masks(
    model: torch.nn.Module,
    temp_files: List[str],
    blk: int,
    hparams: Any,
    config: Any,
    mask_metadata: Dict[str, Tuple[int, str]]
) -> None:
    """
    Process downloaded mask files and apply them to the model.

    Args:
        model (torch.nn.Module): The model to update with averaged masks.
        temp_files (List[str]): List of file paths to the downloaded mask files.
        blk (int): Block number used as a seed for mask generation.
        hparams (Any): Hyperparameters object containing model configurations.
        config (Any): Configuration object containing training settings.
        mask_metadata (Dict[str, Tuple[int, str]]): Mapping from temp file paths to (uid, hotkey).

    Returns:
        None

    Example Usage:
        process_downloaded_masks(model, temp_files, blk, hparams, config, mask_metadata)

    Notes:
        - The function assumes that the same random seed (blk) is used during mask creation and processing.
        - Ensure that mask generation logic is consistent between uploading and processing.
    """
    mask_count: int = 0  # Counter for valid masks processed
    aggregated_masks: Dict[str, torch.Tensor] = {}  # Dictionary to hold aggregated mask values
    mask_indices: Dict[str, torch.Tensor] = {}  # Indices where mask is applied

    # Set random seed for reproducibility
    torch.manual_seed(blk)

    # Generate mask indices for model parameters using the same seed as during mask creation
    for name, param in model.named_parameters():
        param = param.to(config.device)
        # Create a random mask based on the compression ratio
        random_mask = (torch.rand(param.shape, device=config.device) < (1 / hparams.compression)).float()
        indices = random_mask.flatten().nonzero(as_tuple=False).flatten()
        mask_indices[name] = indices

    # Process each downloaded mask file
    for file in temp_files:
        # Retrieve UID and hotkey from mask metadata for debugging
        uid, hotkey = mask_metadata.get(file, (None, None))
        print(f"Processing mask from UID: {uid}, Hotkey: {hotkey}, File: {file}")

        try:
            # Load the mask file
            mask = torch.load(file, map_location='cpu', weights_only=True)
        except Exception as e:
            print(f"Error loading mask file {file} from UID {uid}, Hotkey {hotkey}: {e}")
            continue  # Skip to the next mask file

        mask_count += 1  # Increment valid mask count

        # Iterate over each parameter in the mask
        for name in mask.keys():
            mask_values = mask[name]['values']
            if torch.isnan(mask_values).any():
                print(f"Mask contains NaNs in parameter: {name}, UID: {uid}, Hotkey: {hotkey}")
                continue  # Skip parameters with NaN values

            param_shape = model.get_parameter(name).shape
            indices = mask_indices.get(name)

            # Check for shape mismatch between mask values and indices
            if len(mask_values) != len(indices):
                print(f"Shape mismatch for parameter: {name}, UID: {uid}, Hotkey: {hotkey}")
                print(f"mask_values length: {len(mask_values)}, indices length: {len(indices)}")
                continue  # Skip this parameter due to shape mismatch

            # Decompress the mask values into the full parameter shape
            decompressed = torch.zeros(param_shape, device='cpu').flatten()
            decompressed[indices] = mask_values

            # Aggregate the mask values
            if name not in aggregated_masks:
                aggregated_masks[name] = decompressed.view(param_shape)
            else:
                aggregated_masks[name] += decompressed.view(param_shape)

    if mask_count == 0:
        print("No valid masks processed.")
        return

    # Average the aggregated masks
    for key in aggregated_masks.keys():
        aggregated_masks[key] /= mask_count

    # Update model parameters with averaged masks
    for name, param in model.named_parameters():
        indices = mask_indices.get(name)
        if name in aggregated_masks:
            if aggregated_masks[name].shape == param.shape:
                # Apply the averaged mask to the flattened parameter data
                aggregated_values = aggregated_masks[name].to(model.device).flatten()
                param_flat = param.data.flatten()
                param_flat[indices] = aggregated_values[indices]
                param.data.copy_(param_flat.view(param.shape))
                del aggregated_values, param_flat
            else:
                print(f"Shape mismatch for parameter {name} during assignment, UID: {uid}, Hotkey: {hotkey}")
                print(f"Expected shape: {param.shape}, Received shape: {aggregated_masks[name].shape}")

    # Clean up GPU memory
    for key in aggregated_masks.keys():
        aggregated_masks[key] = aggregated_masks[key].cpu()
    for key in mask_indices.keys():
        mask_indices[key] = mask_indices[key].cpu()
    del mask_indices, aggregated_masks

def create_upload_mask(model: torch.nn.Module, next_upload_block: int, hparams: Any) -> Dict[str, Any]:
    """
    Create a masked state dictionary of the model for uploading.

    Args:
        model (torch.nn.Module): The model to mask.
        next_upload_block (int): The block number for seeding the mask.
        hparams (Any): Hyperparameters object.

    Returns:
        Dict[str, Any]: The masked state dictionary.
    """
    upload_mask = {}
    torch.manual_seed(next_upload_block)
    for name, param in model.named_parameters():
        param = param.to(config.device)
        next_mask = (torch.rand(param.shape) < (1 / hparams.compression)).float()
        upload_mask[name] = next_mask
    model_state_dict = {}
    for name, param in model.named_parameters():
        param_mask = upload_mask[name]
        param_flat = param.flatten()
        mask_flat = param_mask.flatten()
        unmasked_indices = mask_flat.nonzero(as_tuple=False).flatten()
        unmasked_params = param_flat[unmasked_indices]
        # Store the values and indices
        model_state_dict[name] = {
            'values': unmasked_params.to('cpu'),  # Transfer to CPU for saving
            'indices': unmasked_indices.to('cpu')
        }

    return model_state_dict


def upload_mask_async(model_state_dict: Dict[str, Any], hotkey_address: str, next_upload_block: int, config: Any, client: Any, upload_history: List[str]):
    """
    Upload the masked state dictionary to S3 asynchronously.

    Args:
        model_state_dict (Dict[str, Any]): The masked state dictionary.
        hotkey_address (str): The hotkey address of the miner.
        next_upload_block (int): The block number for the upload.
        config (Any): Configuration object.
        client (Any): The S3 client.
        upload_history (List[str]): History of uploaded filenames.

    Returns:
        None
    """
    upload_filename = f'mask-{hotkey_address}-{next_upload_block}.pt'
    with io.BytesIO() as module_buffer:
        torch.save(model_state_dict, module_buffer)
        module_buffer.seek(0)
        client.upload_fileobj(module_buffer, config.bucket, upload_filename)
    client.put_object_acl(
        Bucket=config.bucket,
        Key=upload_filename,
        GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
        GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
    )
    upload_history.append(upload_filename)
    # Delete old mask files to maintain cache size.
    if len(upload_history) > 5:
        to_delete = upload_history.pop(0)
        client.delete_object(Bucket=config.bucket, Key=to_delete)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Miner script optimized for asynchronous operations.')
    parser.add_argument('--name', type=str, default=None, help='Optional miner name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network UID.')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--desired_batch_size', type=int, default=512, help='Training batch size per step')
    parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.95, help='Beta2 for the optimizer')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.1, help='Weight decay for the optimizer')
    parser.add_argument('--grad_clip', type=float, default=None, help='Maximum gradient norm for clipping')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.network = 'test'
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
    main(config)
