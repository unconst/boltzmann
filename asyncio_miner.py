import io
import os
import uuid
import time
import torch
import boto3
import hashlib
import botocore
import tempfile
import argparse
import traceback
import numpy as np
import asyncio
import aioboto3
import wandb
import bittensor as bt
from tqdm import tqdm
from hparams import load_hparams
from dataset import SubsetFineWebEdu2Loader
from functools import lru_cache
from typing import List, Tuple
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import CosineAnnealingLR

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
client_config = botocore.config.Config(
    max_pool_connections=256
)

# Define the asynchronous S3 client
async def get_s3_client():
    session = aioboto3.Session()
    client = await session.client(
        's3',
        region_name='us-east-1',  # AWS region.
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ).__aenter__()
    return client

async def main(config):
    # Print the configuration settings.
    print('\n', '-' * 40, 'Config', '-' * 40)
    print(config)

    # Init Bittensor objects.
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    bt.logging.off()
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    print('\n', '-' * 40, 'Objects', '-' * 40)
    print(f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}')

    # Init my bucket information by submitting it to the chain.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception:
        # If not committed or mismatch, commit the bucket to the chain.
        subtensor.commit(wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket)

    # Initialize Weights and Biases (wandb) for experiment tracking if enabled.
    if config.use_wandb:
        # Check for existing runs with the same name and delete them
        api = wandb.Api()
        runs = api.runs(path="220")
        for run in runs:
            if run.name == f'M{my_uid}':
                print(f'Deleting old run: {run}')
                run.delete()
        wandb.init(project='220', resume='allow', name=f'M{my_uid}', config=config)

    # Init training state.
    print('\n', '-' * 40, 'Hparams', '-' * 40)
    hparams = load_hparams()
    print(hparams)
    model = LlamaForCausalLM(config=hparams.model_config)
    names_and_sizes = [(name, param.numel(), max(1, int(param.numel() // hparams.compression)))
                       for name, param in sorted(model.named_parameters())]

    already_seen_masks = []
    upload_history = []
    last_mask_sync = 0
    last_master_sync = 0
    n_steps = 0

    s3_client = await get_s3_client()

    while True:
        try:
            print('\n', '-' * 40, f'Global Step: {n_steps}', '-' * 40)
            # Start timing for the entire step
            global_step_start_time = time.time()
            n_steps += 1

            # Load hparams.
            # Only sync chain state every 5 steps.
            if n_steps % 5 == 0:
                print(f'\nLoading chain state on step {n_steps} ...')
                load_chain_state_start_time = time.time()
                hparams = load_hparams()
                subtensor = bt.subtensor(config=config)
                metagraph = subtensor.metagraph(netuid=config.netuid)
                print(f'\tLoading chain state completed in {time.time() - load_chain_state_start_time} seconds')

            # Sync the full model state every hparams.epoch_length
            if model is None or subtensor.block - last_master_sync > hparams.epoch_length:
                print(f'\nLoading master state ...')
                load_master_state_start_time = time.time()
                try:
                    master_uid = int(metagraph.S.argmax())
                    master_bucket = 'aladdinformalised'  # Placeholder bucket name
                    master_hotkey = metagraph.hotkeys[master_uid]
                    master_filename = f'master-5GvKEoc787uDV8etY1AM8vF385edu2iyqD1WfCjDugzLUiAL.pt'
                    unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                    await s3_client.download_file(master_bucket, master_filename, unique_temp_file)
                    master_state_dict = torch.load(unique_temp_file, map_location='cpu', weights_only=True)
                    model = LlamaForCausalLM(config=hparams.model_config)
                    model.load_state_dict(master_state_dict)
                    model.to(config.device)
                    model.train()
                    last_master_sync = subtensor.block
                    last_mask_sync = last_master_sync
                except Exception as e:
                    print(f'No master: {e} Waiting ...')
                    await asyncio.sleep(12)
                    continue
                print(f'\tLoading master state completed in {time.time() - load_master_state_start_time} seconds')

            # Reset the optimizer if we need to.
            if (
                'optimizer' not in locals() or
                optimizer is None or
                prev_learning_rate != hparams.learning_rate or
                prev_optimizer_beta1 != hparams.optimizer_beta1 or
                prev_optimizer_beta2 != hparams.optimizer_beta2 or
                prev_optimizer_weight_decay != hparams.optimizer_weight_decay or
                prev_cosine_epoch_length != hparams.cosine_epoch_length or
                prev_eta_min != hparams.eta_min
            ):
                print(f'\nResetting optimizer ...')
                reset_optimizer_start_time = time.time()
                prev_learning_rate = hparams.learning_rate
                prev_optimizer_beta1 = hparams.optimizer_beta1
                prev_optimizer_beta2 = hparams.optimizer_beta2
                prev_optimizer_weight_decay = hparams.optimizer_weight_decay
                prev_cosine_epoch_length = hparams.cosine_epoch_length
                prev_eta_min = hparams.eta_min
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=hparams.learning_rate,  # Peak learning rate
                    betas=(hparams.optimizer_beta1, hparams.optimizer_beta2),  # B1 and B2
                    weight_decay=hparams.optimizer_weight_decay,  # Weight decay
                    foreach=True,  # more memory usage, but faster
                )
                scheduler = CosineAnnealingLR(optimizer, T_max=hparams.cosine_epoch_length, eta_min=hparams.eta_min, last_epoch=-1)
                print(f'\tResetting optimizer completed in {time.time() - reset_optimizer_start_time} seconds')

            print(f'\nGetting blocks and buckets ...')
            get_blocks_and_buckets_start_time = time.time()  # Start timing

            def block_to_mask_window_id(block: int) -> int:
                return int(block / hparams.mask_window_length)

            block = subtensor.block
            all_sync_blocks = list(range(last_mask_sync - 2, block + 1))
            last_mask_sync = block
            # Get buckets per uid if needs update.
            if 'buckets' not in locals() or len(buckets) != len(metagraph.uids):
                buckets = []
                for uid in metagraph.uids:
                    try:
                        buckets.append(subtensor.get_commitment(config.netuid, uid))
                    except:
                        buckets.append(None)
            print(f'\tGetting block completed in {time.time() - get_blocks_and_buckets_start_time} seconds')

            # For each bucket, get all files that need to be synced.
            num_valid_masks = 0
            failed_buckets = 0
            failed_file_masks = 0
            get_masks_names_start_time = time.time()
            mask_filenames_per_mask_wid = {int(block_to_mask_window_id(blk)): [] for blk in all_sync_blocks}
            all_mask_wids = set(list(mask_filenames_per_mask_wid.keys()))
            print(f'\nGetting masks names for blocks: {all_sync_blocks}, windows: {list(mask_filenames_per_mask_wid.keys())} and buckets: {set(buckets)}')

            async def list_masks(bucket):
                masks = []
                try:
                    paginator = s3_client.get_paginator('list_objects_v2')
                    async for page in paginator.paginate(Bucket=bucket, Prefix='mask-'):
                        if 'Contents' not in page:
                            continue
                        for obj in page.get('Contents', []):
                            try:
                                filename = obj['Key']
                                parts = filename.split('-')
                                hotkey = parts[1]
                                mask_wid = int(parts[2].split('.')[0])
                                if hotkey not in metagraph.hotkeys:
                                    continue  # Miner is not registered on network.
                                elif filename in already_seen_masks:
                                    continue
                                elif mask_wid not in all_mask_wids:
                                    continue
                                else:
                                    uid = metagraph.hotkeys.index(hotkey)
                                    mask_info = SimpleNamespace(bucket=bucket, hotkey=hotkey, filename=filename, uid=uid, block=-1, mask_wid=int(mask_wid))
                                    masks.append(mask_info)
                                    already_seen_masks.append(mask_info.filename)
                            except Exception as e:
                                print(f'Error getting mask file with error: {e} for filename: {filename}')
                                continue
                except Exception as e:
                    print(f'\tFailed listing objects in bucket: {bucket} with error: {e}')
                    return []
                return masks

            # Collect masks asynchronously
            tasks = [list_masks(bucket) for bucket in set(buckets) if bucket is not None]
            results = await asyncio.gather(*tasks)
            for masks in results:
                for mask_info in masks:
                    mask_filenames_per_mask_wid[mask_info.mask_wid].append(mask_info)
                    num_valid_masks += 1

            print(f'\tGetting masks: {num_valid_masks} masks for buckets completed in {time.time() - get_masks_names_start_time} seconds')

            # Clean history for memory reasons.
            if len(already_seen_masks) > 256:
                already_seen_masks = already_seen_masks[-256:]

            # Get the mask for mask_wids.
            print(f'\nDownloading {num_valid_masks} masks for: {all_sync_blocks}')
            download_masks_start_time = time.time()
            full_sync_start_time = time.time()
            mask_count_per_id = {}

            async def process_mask_wid(mask_wid):
                # Get the number of masks for this step.
                num_masks_for_mask_wid = len(mask_filenames_per_mask_wid[mask_wid])
                if num_masks_for_mask_wid == 0:
                    return

                # Download the masks from all valid files
                print(f'\n\tDownloading {num_masks_for_mask_wid} mask(s) for mask_wid: {mask_wid} ... ')
                temp_files = []
                n_downloaded = 0
                failed_downloaded = 0

                async def download_file(mask_info):
                    try:
                        temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                        await s3_client.download_file(mask_info.bucket, mask_info.filename, temp_file)
                        mask_info = SimpleNamespace(**vars(mask_info), temp_file=temp_file)
                        return mask_info
                    except Exception as e:
                        print(f'Error downloading file {mask_info.filename} from bucket {mask_info.bucket}: {e}')
                        return None

                download_start_time = time.time()
                download_tasks = [download_file(mask_info) for mask_info in mask_filenames_per_mask_wid[mask_wid]]
                download_results = await asyncio.gather(*download_tasks)

                for result in download_results:
                    if result:
                        temp_files.append(result)
                        n_downloaded += 1
                    else:
                        failed_downloaded += 1
                download_end_time = time.time()
                print(f'\t\tDownloading {n_downloaded}/{n_downloaded + failed_downloaded} masks completed in {download_end_time - download_start_time} seconds')

                # Break the loop when there is nothing to download.
                if n_downloaded == 0:
                    return

                # Get or create the mask for the window.
                create_mask_start_time = time.time()
                mask_indices = {}
                mask_seed_rng = int(hashlib.md5(str(mask_wid).encode('utf-8')).hexdigest(), 16) % (2 ** 32)
                rng = np.random.default_rng(mask_seed_rng)
                print(f'\n\tCreating mask for mask_wid: {mask_wid} and rng: {mask_seed_rng} and compression: {hparams.compression} ...')
                for name_l, param_size_l, num_indices_l in names_and_sizes:
                    indices_l = rng.choice(param_size_l, size=num_indices_l, replace=False)
                    mask_indices[name_l] = torch.from_numpy(indices_l).long().cpu()
                # Compute a hash of all the sizes of all the mask_indices
                sizes_hash = hashlib.md5(str([str(indices.shape) for indices in mask_indices.values() if indices.numel() > 0]).encode('utf-8')).hexdigest()
                print(f'\t\tHash: {sizes_hash}')
                print(f'\t\tCreating mask completed in {time.time() - create_mask_start_time} seconds')

                # Load all masks as state dicts.
                print(f'\n\tLoading state dicts for mask_wid: {mask_wid} ...')
                load_state_dicts_start_time = time.time()
                mask_count = 0
                masks_failed = 0
                mask_successes = 0
                for info in temp_files:
                    try:
                        mask_count += 1
                        mask = torch.load(info.temp_file, map_location=torch.device(config.device), weights_only=True)
                        for name_k, _, _ in names_and_sizes:
                            param_k = model.get_parameter(name_k)
                            values_k = mask[name_k].to(config.device)
                            indices_k = mask_indices[name_k].to(config.device)
                            param_k.data.view(-1)[indices_k] += values_k  # Add the masked values to the local for averaging later.
                            del values_k
                        mask_successes += 1
                    except Exception as e:
                        print(f'Loading mask {info} failed with error: {e}')
                        masks_failed += 1
                        pass
                mask_count_per_id[mask_wid] = mask_count
                if config.use_wandb:
                    wandb.log({"mask_success_rate": (mask_successes) / (mask_successes + masks_failed)})
                print(f'\t\tLoading {mask_successes}/{mask_successes + masks_failed} state dicts completed in {time.time() - load_state_dicts_start_time} seconds')

                # Average the values under the mask.
                print(f'\n\tAveraging {mask_successes} successful masks for mask_wid: {mask_wid} ...')
                average_masks_start_time = time.time()
                for name, param in model.named_parameters():
                    indices = mask_indices[name].to(config.device)
                    param.data.view(-1)[indices] /= (mask_successes + 1)  # Average (only) the masked values
                print(f'\t\tAveraged state dicts in {time.time() - average_masks_start_time} seconds')

                print(f'\n\tDeleting files for mask_wid: {mask_wid} ...')
                del mask_indices
                delete_files_start_time = time.time()
                for info in temp_files:
                    os.remove(info.temp_file)
                print(f'\t\tDeleting files completed in {time.time() - delete_files_start_time} seconds')

            process_tasks = [process_mask_wid(mask_wid) for mask_wid in mask_filenames_per_mask_wid.keys()]
            await asyncio.gather(*process_tasks)

            # Log the average number of masks applied per mask_wid
            avg_masks_per_mask_wid = sum(mask_count_per_id.values()) / len(mask_count_per_id) if mask_count_per_id else 0
            if config.use_wandb:
                wandb.log({"avg_masks_per_mask_wid": avg_masks_per_mask_wid})

            # Print completion
            print(f'\nDownloading masks for blocks: {all_sync_blocks} and mask_wids: {list(mask_filenames_per_mask_wid.keys())} in {time.time() - download_masks_start_time} seconds')
            del mask_filenames_per_mask_wid
            torch.cuda.empty_cache()

            # Get the pages for this block and my_uid.
            # This is global and deterministic
            n_pages = max(1, int(hparams.desired_batch_size * 0.01))
            print(f'\nLoading {n_pages} pages ...')
            load_pages_start_time = time.time()  # Start timing
            pages = SubsetFineWebEdu2Loader.next_pages(
                offset=subtensor.block + hparams.pages_window_speed,
                n_pages=n_pages,
                seed=my_uid
            )
            dataset = SubsetFineWebEdu2Loader(
                batch_size=config.actual_batch_size,
                sequence_length=hparams.sequence_length,
                pages_info=pages,
                tokenizer=hparams.tokenizer
            )
            print(f'\tLoading {n_pages} pages completed in {time.time() - load_pages_start_time} seconds')

            # Train my model on the current page.
            print(f'\nTraining {n_pages} pages ...')
            train_pages_start_time = time.time()
            torch.cuda.empty_cache()  # Empty cache going into the training step.
            optimizer.zero_grad()  # Clear any lingering grads.
            start_time = time.time()  # Start timing
            total_loss = 0.0
            total_steps = hparams.desired_batch_size // config.actual_batch_size
            for idx, batch in enumerate(dataset):
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                with torch.amp.autocast(device_type=model.device.type, dtype=torch.bfloat16):  # Enable autocasting for mixed precision
                    outputs = model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                loss = outputs.loss / (total_steps + 1)  # Divide by number of accumulations.
                loss.backward()
                if idx >= total_steps - 1:
                    break

            # Try step with error handling.
            try:
                # grad norm clipping
                if hparams.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
                optimizer.step()
                scheduler.step()  # Update the learning rate.
                optimizer.zero_grad()

                # Clean lingering objects
                del input_ids, labels, outputs
                torch.cuda.empty_cache()  # Empty cache at end of step.
            except AssertionError as e:
                print(f"\tAn error occurred during the optimizer step: {e}")

            # Calculate, print and log average loss
            average_loss = total_loss / total_steps
            total_time = time.time() - train_pages_start_time
            steps_per_second = total_steps / total_time
            batches_per_second = config.actual_batch_size * total_steps / total_time
            tokens_per_second = hparams.sequence_length * config.actual_batch_size * total_steps / total_time
            if config.use_wandb:
                wandb.log({
                    "step_loss": average_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "incentive": float(metagraph.I[my_uid]),
                    "steps_per_second": steps_per_second,
                    "batches_per_second": batches_per_second,
                    "tokens_per_second": tokens_per_second
                })
            print('\tloss:', average_loss, 'learning_rate:', scheduler.get_last_lr()[0])
            print(f'\tTraining completed in {total_time} seconds, Steps per second: {steps_per_second}, Batches per second: {batches_per_second}, Tokens per second: {tokens_per_second}')

            # Select the block to produce a mask for.
            next_upload_block = subtensor.block

            # Get the proper mask for my upload block + page.
            mask_seed = int(block_to_mask_window_id(next_upload_block))
            create_upload_mask_start_time = time.time()
            mask_indices = {}
            mask_seed_rng = int(hashlib.md5(str(mask_seed).encode('utf-8')).hexdigest(), 16) % (2 ** 32)
            rng = np.random.default_rng(mask_seed_rng)
            print(f'\nCreating mask for mask_wid: {mask_seed} and rng: {mask_seed_rng} and compression: {hparams.compression} ...')
            for name_j, param_size_j, num_indices_j in names_and_sizes:
                indices_j = rng.choice(param_size_j, size=num_indices_j, replace=False)
                indices_tensor = torch.from_numpy(indices_j).long().cpu()
                mask_indices[name_j] = indices_tensor
            sizes_hash = hashlib.md5(str([str(indices.shape) for indices in mask_indices.values() if indices.numel() > 0]).encode('utf-8')).hexdigest()
            print(f'\t\tHash: {sizes_hash}')
            print(f'\tCreating upload mask completed in {time.time() - create_upload_mask_start_time} seconds')

            # Mask the model values given the mask and produce a state dict.
            print(f'\nApply {mask_seed} upload mask to model ...')
            apply_upload_mask_start_time = time.time()
            model_state_dict = model.state_dict()
            for name_i, _, _ in names_and_sizes:
                param_i = model.get_parameter(name_i)
                model_state_dict[name_i] = param_i.flatten()[mask_indices[name_i].to(config.device)].cpu()
            del mask_indices
            print(f'\tApplied mask to model completed in: {time.time() - apply_upload_mask_start_time} seconds')

            # Upload the state dict of my masked weights.
            print(f'\nUploading mask for block:{next_upload_block} in mask window: {mask_seed}...')
            upload_mask_start_time = time.time()
            upload_filename = f'mask-{wallet.hotkey.ss58_address}-{mask_seed}.pt'
            with io.BytesIO() as module_buffer:
                torch.save(model_state_dict, module_buffer)
                module_buffer.seek(0)  # Reset the buffer's position to the beginning.
                await s3_client.upload_fileobj(module_buffer, config.bucket, upload_filename)
            await s3_client.put_object_acl(
                Bucket=config.bucket,
                Key=upload_filename,
                GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
                GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
            )
            upload_history.append(upload_filename)
            print(f'\tUploading mask to: {upload_filename} completed in {time.time() - upload_mask_start_time} seconds')

            # Delete old mask files and clean.
            print('\nDeleting history ...')
            delete_history_start_time = time.time()
            if len(upload_history) > hparams.max_history:
                to_delete = upload_history.pop(0)
                await s3_client.delete_object(Bucket=config.bucket, Key=to_delete)
            print(f'\tDeleting history completed in {time.time() - delete_history_start_time} seconds')

            # Calculate and log global steps per second
            global_step_total_time = time.time() - global_step_start_time
            global_steps_per_second = 1 / global_step_total_time
            if config.use_wandb:
                wandb.log({
                    "global_steps_per_second": global_steps_per_second,
                    "global_step_time": global_step_total_time,
                    "global_tokens_per_second": hparams.sequence_length * config.actual_batch_size * total_steps / global_step_total_time
                })
            print(f'\nGlobal step completed in {global_step_total_time} seconds\n')

        # Handle keyboard interrupts to allow graceful shutdown.
        except (KeyboardInterrupt, SystemExit):
            # Clean up by deleting the model from S3 if it exists.
            print("Training interrupted. Exiting gracefully.")
            await s3_client.close()
            break

        # Handle any other exceptions, log the error, and continue after a short delay.
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Miner script')
    parser.add_argument('--name', type=str, default=None, help='Optional miner name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network UID.')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    config = bt.config(parser)
    config.subtensor.network = 'test'
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'

    asyncio.run(main(config))
