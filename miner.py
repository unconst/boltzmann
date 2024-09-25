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

import io
import os
import uuid
import time
import wandb
import boto3
import torch
import botocore
import tempfile
import argparse
import traceback
import numpy as np
from tqdm import tqdm
import bittensor as bt
import concurrent.futures  
import torch.optim as optim
from typing import List, Tuple
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import LlamaForCausalLM 
from torch.optim.lr_scheduler import CosineAnnealingLR

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
client_config = botocore.config.Config(
    max_pool_connections=256
)
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',  # AWS region.
    config = client_config,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def main(config):
    # Print the configuration settings.
    print('\n', '-' * 40, 'Config', '-' * 40)
    print(config)
    
    # Init Bittensor objects.
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)    
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
        run = wandb.init(project='cont', resume='allow', name=f'M{my_uid}', config=config)
        
    # Init training state.
    print('\n', '-' * 40, 'Hparams', '-' * 40)
    hparams = load_hparams()
    print ( hparams ) 
    model = None
    already_seen_masks = []
    upload_history = []  
    last_mask_sync = 0 
    last_master_sync = 0
    n_steps = 0
    while True:
        try:   
            print('\n', '-' * 40, f'Step: {n_steps}', '-' * 40) 
            # Start timing for the entire step
            global_step_start_time = time.time()
            n_steps += 1
            
            # Load hparams.
            print ('\nLoading hparams ...')
            start_time = time.time()
            new_hparams = load_hparams()
            hparams_changed = any(getattr(new_hparams, key) != getattr(hparams, key) for key in set(vars(new_hparams)) | set(vars(hparams)))
            hparams = new_hparams
            print(f'\tLoading hparams completed in {time.time() - start_time} seconds') 

            # Sync the current chain state and hparams.
            print ('\nLoading chain state ...')
            start_time = time.time()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
            print(f'\tLoading chain state completed in {time.time() - start_time} seconds') 
            
            # Sync the full model state every hparams.epoch_length
            if model == None or subtensor.block - last_master_sync > hparams.epoch_length:
                print(f'\nLoading master state ...') 
                start_time = time.time() 
                try:
                    master_uid = int(metagraph.S.argmax())
                    master_bucket = 'aladdinformalised' #'#subtensor.get_commitment( config.netuid, master_uid )
                    master_hotkey = metagraph.hotkeys[ master_uid ]
                    master_filename = f'master-5GvKEoc787uDV8etY1AM8vF385edu2iyqD1WfCjDugzLUiAL.pt'
                    unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                    CLIENT.download_file( master_bucket, master_filename, unique_temp_file )
                    master_state_dict = torch.load( unique_temp_file, map_location='cpu', weights_only = True )
                    model = LlamaForCausalLM( config = hparams.model_config )
                    model.load_state_dict( master_state_dict )
                    model.to(config.device)
                    model.train()
                    last_master_sync = subtensor.block 
                    last_mask_sync = last_master_sync
                except Exception as e:
                    print (f'No master:{e} Waiting ...')
                    time.sleep(12)
                    continue
                print(f'\tCLoading master state completed in {time.time() - start_time} seconds') 
            
            if 'optimizer' not in locals() or optimizer == None or hparams_changed:
                print(f'\nResetting optimizer ...') 
                start_time = time.time()
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr = hparams.learning_rate,  # Peak learning rate
                    betas = ( hparams.optimizer_beta1, hparams.optimizer_beta2 ), # B1 and B2
                    weight_decay = hparams.optimizer_weight_decay,  # Weight decay
                    foreach = True,  # more memory usage, but faster
                )
                scheduler = CosineAnnealingLR( optimizer, T_max = hparams.cosine_epoch_length, eta_min=hparams.eta_min, last_epoch=-1 )
                print(f'\tResetting optimizer completed in {time.time() - start_time} seconds') 


            print(f'\nGetting blocks and buckets ...')
            start_time = time.time()  # Start timing
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
            print(f'\tGetting block completed in {time.time() - start_time} seconds')

            # For each bucket, get all files that need to be synced.
            num_valid_masks = 0
            failed_buckets = 0
            failed_file_masks = 0
            start_time = time.time()
            mask_filenames_per_mask_wid = {int(block_to_mask_window_id(blk)): [] for blk in all_sync_blocks}
            all_mask_wids = set(list(mask_filenames_per_mask_wid.keys()))
            print(f'\nGetting masks names for blocks: {all_sync_blocks}, windows: {list(mask_filenames_per_mask_wid.keys())} and buckets: {set(buckets)}')
            for bucket in list(set(buckets)):
                if bucket is None:
                    continue
                try:
                    paginator = CLIENT.get_paginator('list_objects_v2')
                    page_iterator = paginator.paginate(Bucket=bucket, Prefix='mask-')
                    for page in page_iterator:
                        if 'Contents' not in page: continue
                        for obj in page.get('Contents', []):
                            try:
                                filename = obj['Key']
                                hotkey, mask_wid = filename.split('-')[1], filename.split('-')[2].split('.')[0]
                                mask_wid = int(mask_wid)
                                if hotkey not in metagraph.hotkeys: 
                                    failed_file_masks += 1
                                    print (f'Discarding {filename}, {hotkey} not registered.')
                                    continue # Miner is not registered on network.
                                elif filename in already_seen_masks:
                                    failed_file_masks += 1
                                    print (f'Discarding {filename}, because already seen.')
                                    continue
                                elif mask_wid not in all_mask_wids:
                                    failed_file_masks += 1
                                    print (f'Discarding {filename}, because {mask_wid} not in {all_mask_wids}')
                                    continue
                                else:
                                    uid = metagraph.hotkeys.index(hotkey)
                                    mask_info = SimpleNamespace(bucket=bucket, hotkey=hotkey, filename=filename, uid=uid, block=-1, mask_wid=int(mask_wid))
                                    mask_filenames_per_mask_wid[mask_wid].append(mask_info)
                                    already_seen_masks.append( mask_info.filename )
                                    num_valid_masks += 1
                                    print (f'Applying {filename}. Success.')

                            except Exception as e:
                                print (f'Error getting mask file with error: {e} for filename: {filename}')
                                continue
                except Exception as e: 
                    failed_buckets += 1
                    print (f'\tFailed listing objects in bucket: {bucket} with error: {e}')
                    continue
            print(f'\tGetting masks: {num_valid_masks}/{num_valid_masks + failed_file_masks} masks for buckets: {len(buckets) - failed_buckets}/{len(buckets)} for buckets {set(buckets)} completed in {time.time() - start_time} seconds')
            
            # Clean history for memory reasons.
            if len(already_seen_masks) > 256:
                already_seen_masks = already_seen_masks[-256:]

            # Get the mask for mask_wids.
            print(f'\nDownloading {num_valid_masks} masks for: {all_sync_blocks}')
            full_sync_start_time = time.time()
            masks_per_id_per_uid = {}
            mask_count_per_id = {}
            for mask_wid in mask_filenames_per_mask_wid.keys():
                masks_per_id_per_uid[mask_wid] = {}
                # Get the number of masks for this step.
                num_masks_for_mask_wid = len(mask_filenames_per_mask_wid[mask_wid])
                if num_masks_for_mask_wid == 0:
                    continue

                # Download the masks from all valid files
                print(f'\n\tDownloading {num_masks_for_mask_wid} mask for mask_wid: {mask_wid} ... ')
                start_time = time.time()
                temp_files = []
                n_downloaded = 0
                failed_downloaded = 0
                def download_file(mask_info):
                    try:
                        temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                        CLIENT.download_file(mask_info.bucket, mask_info.filename, temp_file)
                        mask_info = SimpleNamespace(**vars(mask_info), temp_file=temp_file)
                        return mask_info
                    except:
                        return None

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(download_file, mask_info) for mask_info in mask_filenames_per_mask_wid[mask_wid]]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            temp_files.append(result)
                            n_downloaded += 1
                        else:
                            failed_downloaded += 1
                print(f'\t\tDownloading {n_downloaded}/{n_downloaded + failed_downloaded} masks completed in {time.time() - start_time} seconds')

                # Break the loop when there is nothing to download.
                if n_downloaded == 0:
                    continue

                # Init the mask indices using the block number.
                print(f'\n\tCreating mask for mask_wid: {mask_wid} ...')
                mask_indices = {}
                np.random.seed(int(mask_wid))
                start_time = time.time()
                for name, param in model.named_parameters():
                    param = param.to(config.device)
                    param_shape = param.shape
                    random_values = np.random.rand(*param_shape)  # Generate NumPy random values in [0, 1)
                    next_mask = (random_values < (1 / hparams.compression)).astype(np.float32)  # Apply compression ratio
                    next_mask_tensor = torch.from_numpy(next_mask).to(config.device)
                    indices = next_mask_tensor.flatten().nonzero(as_tuple=False).flatten()
                    mask_indices[name] = indices
                print(f'\t\tCreating mask completed in {time.time() - start_time} seconds')

                # Load all masks as state dicts.
                print(f'\n\tLoading state dicts for mask_wid: {mask_wid} ...')
                start_time = time.time()
                mask_count = 0
                masks_failed = 0
                mask_successes = 0
                masks_dicts_values = {}
                for info in temp_files:
                    try:
                        masks_per_id_per_uid[info.mask_wid][info.uid] = {}
                        mask = torch.load(info.temp_file, map_location='cpu', weights_only=True)
                        mask_count += 1
                        for name in mask.keys():
                            mask_values = mask[name]['values']
                            if torch.isnan(mask_values).any():
                                continue
                            param_shape = model.get_parameter(name).shape
                            indices = mask_indices[name]
                            decompressed = torch.zeros(param_shape, device='cpu').flatten()
                            decompressed[indices] = mask_values
                            masks_per_id_per_uid[info.mask_wid][info.uid][name] = decompressed.view(param_shape)
                            if name not in masks_dicts_values:
                                masks_dicts_values[name] = decompressed.view(param_shape)
                            else:
                                masks_dicts_values[name] += decompressed.view(param_shape)
                        mask_successes += 1
                    except Exception as e: 
                        print (f'Loading mask {info} failed with error: {e}')
                        masks_failed += 1
                        pass
                mask_count_per_id[mask_wid] = mask_count
                if config.use_wandb: wandb.log({"mask_success_rate": (mask_successes)/(mask_successes + masks_failed)})
                print(f'\t\tLoading {mask_successes}/{mask_successes + masks_failed} state dicts completed in {time.time() - start_time} seconds')
                
                # Check for no masks.
                if mask_successes != 0:
                    # Average the masks before applying.
                    print(f'\n\tAveraging {mask_successes} successful masks for mask_wid: {mask_wid} ...')
                    start_time = time.time()
                    for key in masks_dicts_values.keys():
                        masks_dicts_values[key] /= mask_successes
                    print(f'\t\tAveraged state dicts in {time.time() - start_time} seconds')

                    # Set the average into the model.
                    print(f'\n\tApplying {mask_successes} masks for mask_wid: {mask_wid} ...')
                    start_time = time.time()  # Start timing
                    for name, param in model.named_parameters():
                        indices = mask_indices[name]
                        if name in masks_dicts_values:
                            if masks_dicts_values[name].shape == param.shape:
                                # Apply the mask values to the flattened param data.
                                on_device = masks_dicts_values[name].to(model.device).flatten()
                                param_flat = param.data.flatten()
                                param_flat[indices] = on_device[indices]
                                param.data.copy_(param_flat.view(param.shape))
                                del on_device, param_flat
                            else:
                                print(f"Shape mismatch for {name}: expected {param.shape}, got {masks_dicts_values[name].shape}")
                    for key in masks_dicts_values.keys():
                        masks_dicts_values[key] = masks_dicts_values[key].cpu()
                    for key in mask_indices.keys():
                        mask_indices[key] = mask_indices[key].cpu()
                    del mask_indices, masks_dicts_values
                    print(f'\t\tApplying {mask_count} masks completed in {time.time() - start_time} seconds')
                else:
                    print(f'\t\tNot successful masks added to average.')
                    continue

                # Delete files and clean up.
                print(f'\n\tDeleting files for mask_wid: {mask_wid} ...')
                start_time = time.time()
                for info in temp_files:
                    os.remove(info.temp_file)
                print(f'\t\tDeleting files completed in {time.time() - start_time} seconds')

            # Log the average number of masks applied per mask_wid
            avg_masks_per_mask_wid = sum(mask_count_per_id.values()) / len(mask_count_per_id) if mask_count_per_id else 0
            if config.use_wandb: wandb.log({"avg_masks_per_mask_wid": avg_masks_per_mask_wid})

            # Print completion
            print(f'\tDownloading masks for blocks: {all_sync_blocks} and mask_wids: {list(mask_filenames_per_mask_wid.keys())} in {time.time() - full_sync_start_time} seconds')
            del mask_filenames_per_mask_wid
            torch.cuda.empty_cache()
            
            # Get the pages for this block and my_uid.
            # This is global and deterministic
            n_pages = max(1, int(hparams.desired_batch_size * 0.01))
            print (f'\nLoading {n_pages} pages ...')
            start_time = time.time()  # Start timing
            pages = SubsetFineWebEdu2Loader.next_pages(
                offset = subtensor.block + hparams.pages_window_speed,
                n_pages = n_pages,
                seed = my_uid 
            )
            dataset = SubsetFineWebEdu2Loader(
                batch_size = config.actual_batch_size,
                sequence_length = hparams.sequence_length,
                pages_info = pages,
                tokenizer = hparams.tokenizer
            )
            # TODO: see if wrapping dataloader is faster, with multiple workers and pin_memory=True
            # dataset = torch.utils.data.DataLoader( dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True )
            print(f'\n\tLoading {n_pages} pages completed in {time.time() - start_time} seconds')
            
            # Train my model on the current page.
            print (f'\nTraining {n_pages} pages ...')
            torch.cuda.empty_cache() # Empty cache going into the training step.
            optimizer.zero_grad() # Clear any lingering grads.
            start_time = time.time()  # Start timing
            total_loss = 0.0
            total_steps = hparams.desired_batch_size // config.actual_batch_size
            progress_bar = tqdm(total=total_steps, desc="Training:")
            for idx, batch in enumerate(dataset):
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                with torch.amp.autocast( device_type = model.device.type, dtype = torch.bfloat16 ):  # Enable autocasting for mixed precision
                    outputs = model(input_ids = input_ids, labels=labels)
                total_loss += outputs.loss.item()
                loss = outputs.loss / (total_steps + 1) # Divide by number of accumulations.
                loss.backward()
                progress_bar.update(1)  # Update the progress bar
                if idx >= total_steps - 1:
                    break
            progress_bar.close()  # Close the progress bar
            
            # Try step with error handling.
            try:
                # grad norm clipping
                if hparams.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
                optimizer.step()
                scheduler.step()  # Update the learning rate.
                optimizer.zero_grad()
            except AssertionError as e:
                print(f"An error occurred during the optimizer step: {e}")
            
            # Clean lingering objects
            del input_ids, labels, outputs
            torch.cuda.empty_cache() # Empty cache at end of step.
            
            # Calculate, print and logg average loss
            average_loss = total_loss / total_steps
            total_time = time.time() - start_time
            steps_per_second = total_steps / total_time
            batches_per_second = config.actual_batch_size * total_steps / total_time
            tokens_per_second = hparams.sequence_length * config.actual_batch_size * total_steps / total_time
            if config.use_wandb:
                wandb.log({
                    "step_loss": average_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    f"incentive{my_uid}": float(metagraph.I[my_uid]),
                    "steps_per_second": steps_per_second,
                    "batches_per_second": batches_per_second,
                    "tokens_per_second": tokens_per_second
                })
            print('\tloss:', average_loss, 'learning_rate:', scheduler.get_last_lr()[0])
            print(f'\tTraining completed in {total_time} seconds, Steps per second: {steps_per_second}, Batches per second: {batches_per_second}, Tokens per second: {tokens_per_second}')
            
            # Select the block to produce a mask for.
            next_upload_block = subtensor.block
            
            # Get the proper mask for my upload block + page.
            start_time = time.time()  # Start timing
            upload_mask = {}
            mask_seed = block_to_mask_window_id(next_upload_block)
            print(f'\nCreating upload mask for window: {mask_seed} ...')
            np.random.seed( int(mask_seed) )
            for name, param in model.named_parameters():
                param = param.to(config.device)
                param_shape = param.shape
                random_values = np.random.rand(*param_shape)  # Generate NumPy random values in [0, 1)
                next_mask = (random_values < (1 / hparams.compression)).astype(np.float32)  # Apply compression ratio
                next_mask_tensor = torch.from_numpy(next_mask).to(config.device)
                indices = next_mask_tensor.flatten().nonzero(as_tuple=False).flatten()
                upload_mask[name] = indices
            print(f'\tCreating upload mask_wid mask completed in {time.time() - start_time} seconds')
            
            # Mask the model values given the mask and produce a state dict.                
            print(f'\nApply {mask_seed} upload mask to model ...')
            model_state_dict = model.state_dict()
            for name, param in model.named_parameters():
                param_mask = upload_mask[name].to(param.device)
                param_flat = param.flatten()
                mask_flat = param_mask.flatten()
                unmasked_indices = mask_flat.nonzero(as_tuple=False).flatten()
                unmasked_params = param_flat[unmasked_indices]
                model_state_dict[name] = {'values': unmasked_params.to('cpu')}
                del unmasked_indices
            del upload_mask
            print(f'\tApplied mask to model completed in: {time.time() - start_time} seconds')

            # Upload the state dict of my masked weights.
            print(f'\nUploading mask for block:{next_upload_block} in mask window: {mask_seed}...')
            start_time = time.time()
            upload_filename = f'mask-{wallet.hotkey.ss58_address}-{mask_seed}.pt'
            with io.BytesIO() as module_buffer:
                torch.save(model_state_dict, module_buffer)
                module_buffer.seek(0)  # Reset the buffer's position to the beginning.
                CLIENT.upload_fileobj(module_buffer, config.bucket, upload_filename)
            CLIENT.put_object_acl(
                Bucket=config.bucket,
                Key=upload_filename,
                GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
                GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
            )
            upload_history.append(upload_filename)
            print(f'\tUploading mask to: {upload_filename} completed in {time.time() - start_time} seconds')

            # Delete old mask files and clean.
            print('\nDeleting history ...')
            start_time = time.time()
            if len(upload_history) > hparams.max_history:
                to_delete = upload_history.pop(0)
                CLIENT.delete_object(Bucket=config.bucket, Key=to_delete)
            print(f'\tDeleting history completed in {time.time() - start_time} seconds')
            
            # Calculate and log global steps per second
            global_step_total_time = time.time() - global_step_start_time
            global_steps_per_second = 1 / global_step_total_time
            if config.use_wandb:
                wandb.log({
                    "global_steps_per_second": global_steps_per_second
                })
                 
        # Handle keyboard interrupts to allow graceful shutdown.
        except (KeyboardInterrupt, SystemExit):
            # Clean up by deleting the model from S3 if it exists.
            print("Training interrupted. Exiting gracefully.")
            break
    
        # Handle any other exceptions, log the error, and continue after a short delay.
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(5)
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
    main(config)
