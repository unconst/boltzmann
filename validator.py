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
import random
import hashlib
import botocore
import tempfile
import argparse
import traceback
import numpy as np
from tqdm import tqdm
import bittensor as bt
import concurrent.futures  
import torch.optim as optim
from functools import lru_cache
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
        try:
            runs = api.runs(path=config.project)
            for run in runs:
                if run.name == f'V{my_uid}':
                    print(f'Deleting old run: {run}')
                    run.delete()
        except: pass
        run = wandb.init(project=config.project, resume='allow', name=f'V{my_uid}', config=config)
        
    # Init training state.
    print('\n', '-' * 40, 'Hparams', '-' * 40)
    hparams = load_hparams()
    print ( hparams ) 
    model = LlamaForCausalLM( config = hparams.model_config )
        
    already_seen_masks = []
    upload_history = []  
    last_mask_sync = 0 
    last_master_sync = 0
    global_step = 0
    weights = torch.zeros(metagraph.n, dtype=torch.float32)
    while True:
        try:   
            print('\n', '-' * 40, f'Global Step: {global_step}', '-' * 40) 
            # Start timing for the entire step
            global_step_start_time = time.time()
            global_step += 1
            
            # Load hparams.
            # Only sync chain state every 5 steps.
            if global_step % 5 == 0:
                print (f'\nLoading chain state on step {global_step} ...')
                load_chain_state_start_time = time.time()
                hparams = load_hparams()
                subtensor = bt.subtensor(config=config)
                metagraph = subtensor.metagraph(netuid=config.netuid)
                weights = torch.cat((weights, torch.zeros(metagraph.n - weights.shape[0], dtype=torch.float32))) if weights.shape[0] != metagraph.n else weights
                print(f'\tLoading chain state completed in {time.time() - load_chain_state_start_time} seconds') 
            
            # Sync the full model state every hparams.epoch_length
            if model is None or subtensor.block - last_master_sync > hparams.epoch_length:
                print(f'\nLoading master state ...') 
                load_master_state_start_time = time.time() 
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
                    model.eval()
                    last_master_sync = subtensor.block 
                    last_mask_sync = last_master_sync
                except Exception as e:
                    print (f'No master:{e} Waiting ...')
                    time.sleep(12)
                    continue
                print(f'\tLoading master state completed in {time.time() - load_master_state_start_time} seconds') 
            

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
                for uid in tqdm(metagraph.uids):
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
                                parts = filename.split('-')
                                hotkey = parts[1]
                                mask_wid = int(parts[2].split('.')[0])
                                if hotkey not in metagraph.hotkeys: failed_file_masks += 1; continue # Miner is not registered on network.
                                elif filename in already_seen_masks: failed_file_masks += 1; continue
                                elif mask_wid not in all_mask_wids: failed_file_masks += 1; continue
                                else:
                                    uid = metagraph.hotkeys.index(hotkey)
                                    mask_info = SimpleNamespace(bucket=bucket, hotkey=hotkey, filename=filename, uid=uid, block=-1, mask_wid=int(mask_wid))
                                    mask_filenames_per_mask_wid[mask_wid].append(mask_info)
                                    already_seen_masks.append( mask_info.filename )
                                    num_valid_masks += 1
                                    print (f'\t - Applying uid:{uid} {filename}@{bucket}. Success.')

                            except Exception as e:
                                print (f'Error getting mask file with error: {e} for filename: {filename}')
                                continue
                except Exception as e: 
                    failed_buckets += 1
                    print (f'\tFailed listing objects in bucket: {bucket} with error: {e}')
                    continue
            print(f'\tGetting masks: {num_valid_masks}/{num_valid_masks + failed_file_masks} masks for buckets: {len(buckets) - failed_buckets}/{len(buckets)} for buckets {set(buckets)} completed in {time.time() - get_masks_names_start_time} seconds')
            
            # Clean history for memory reasons.
            if len(already_seen_masks) > 256:
                already_seen_masks = already_seen_masks[-256:]

            # Get the mask for mask_wids.
            print(f'\nDownloading {num_valid_masks} masks for: {all_sync_blocks}')
            download_masks_start_time = time.time()
            full_sync_start_time = time.time()
            mask_count_per_id = {}
            for mask_wid in mask_filenames_per_mask_wid.keys():
                # Get the number of masks for this step.
                num_masks_for_mask_wid = len(mask_filenames_per_mask_wid[mask_wid])
                if num_masks_for_mask_wid == 0:
                    continue

                # Download the masks from all valid files
                print(f'\n\tDownloading {num_masks_for_mask_wid} mask for mask_wid: {mask_wid} ... ')
                download_file_start_time = time.time()
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

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(mask_filenames_per_mask_wid[mask_wid])) as executor:
                    futures = [executor.submit(download_file, mask_info) for mask_info in mask_filenames_per_mask_wid[mask_wid]]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            temp_files.append(result)
                            n_downloaded += 1
                        else:
                            failed_downloaded += 1
                print(f'\t\tDownloading {n_downloaded}/{n_downloaded + failed_downloaded} masks completed in {time.time() - download_file_start_time} seconds')

                # Break the loop when there is nothing to download.
                if n_downloaded == 0:
                    continue

                # Get or create the mask for the window.
                create_mask_start_time = time.time()
                mask_indices = {}
                mask_wid_rng = int(hashlib.md5(str(mask_wid).encode('utf-8')).hexdigest(), 16) % (2**32)
                rng = np.random.default_rng( int(hashlib.md5(str(mask_wid).encode('utf-8')).hexdigest(), 16) % (2**32) )
                print(f'\n\tCreating mask for mask_wid: {mask_wid} and rng: {mask_wid_rng} and compression: {hparams.compression} ...')
                for name, param in sorted( model.named_parameters() ):
                    indices = rng.choice( param.numel(), size = max( 1, int( param.numel() // hparams.compression ) ), replace=False)
                    mask_indices[ name ] = torch.from_numpy( indices ).long().cpu()
                print(f'\t\tCreating mask completed in {time.time() - create_mask_start_time} seconds')

                # Load all masks as state dicts.
                print(f'\n\tLoading state dicts for mask_wid: {mask_wid} ...')
                load_state_dicts_start_time = time.time()
                mask_count = 0
                masks_failed = 0
                mask_successes = 0
                mask_successes_per_param = {name: 0 for name, _ in model.named_parameters()}
                for info in temp_files:
                    try:
                        mask_count += 1
                        mask = torch.load(info.temp_file, map_location=torch.device(model.device), weights_only=True)
                        for name, param in sorted(model.named_parameters()):
                            values = mask[name].to(model.device)
                            indices = mask_indices[name].to(model.device)
                            param.data.view(-1)[indices] += values  # Add the masked values to the local for averaging later.
                            mask_successes_per_param[name] += 1
                            del values
                        mask_successes += 1
                    except Exception as e:
                        print(f'Loading mask {info} failed with error: {e} -- name: {name} -- values: {values.shape} -- indices: {indices.shape} -- param: {param.shape}')
                        masks_failed += 1
                        pass
                mask_count_per_id[mask_wid] = mask_count
                if config.use_wandb: wandb.log({"mask_success_rate": (mask_successes)/(mask_successes + masks_failed)})
                print(f'\t\tLoading {mask_successes}/{mask_successes + masks_failed} state dicts completed in {time.time() - load_state_dicts_start_time} seconds')
                
                # Average the values under the mask.
                print(f'\n\tAveraging {mask_successes} successful masks for mask_wid: {mask_wid} ...')
                average_masks_start_time = time.time()
                for name, param in model.named_parameters():
                    indices = mask_indices[name].to(config.device)
                    param.data.view(-1)[indices] /= (mask_successes_per_param[name] + 1)  # Average (only) the masked values
                print(f'\t\tAveraged state dicts in {time.time() - average_masks_start_time} seconds')

                print(f'\n\tDeleting files for mask_wid: {mask_wid} ...')
                del mask_indices
                delete_files_start_time = time.time()
                for info in temp_files:
                    os.remove(info.temp_file)
                print(f'\t\tDeleting files completed in {time.time() - delete_files_start_time} seconds')

            # Log the average number of masks applied per mask_wid
            avg_masks_per_mask_wid = sum(mask_count_per_id.values()) / len(mask_count_per_id) if mask_count_per_id else 0
            if config.use_wandb: wandb.log({"avg_masks_per_mask_wid": avg_masks_per_mask_wid})

            # Print completion
            print(f'\nDownloading masks for blocks: {all_sync_blocks} and mask_wids: {list(mask_filenames_per_mask_wid.keys())} in {time.time() - download_masks_start_time} seconds')
            
            torch.cuda.empty_cache()

            # Get a random mask to eval.
            print(f'\nEvaling slices.')
            mask_wid = max( list(mask_filenames_per_mask_wid.keys()) )
            n_evals = 0
            while n_evals < hparams.validator_evals_per_step:
                
                # randomly select a miner to eval.
                n_evals += 1
                miner_uid = random.choice( list(metagraph.uids) )
                
                try:
                    # Get the mask implied for this window.         
                    print(f'\nLoading miner mask for uid: {miner_uid}')
                    load_mask_start_time = time.time()
                    miner_bucket = subtensor.get_commitment(config.netuid, miner_uid)    
                    miner_mask_filename = f"mask-{metagraph.hotkeys[miner_uid]}-{mask_wid}.pt"   
                    mask_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                    CLIENT.download_file(miner_bucket, miner_mask_filename, mask_temp_file)
                    mask_values = torch.load( mask_temp_file, map_location = torch.device(model.device), weights_only=True )
                    load_mask_end_time = time.time()
                    print(f'\t\tLoading miner mask completed in {load_mask_end_time - load_mask_start_time} seconds')
                    
                    # Create the mask for the window.
                    create_mask_start_time = time.time()
                    mask_indices = {}
                    mask_wid_rng = int(hashlib.md5(str(mask_wid).encode('utf-8')).hexdigest(), 16) % (2**32)
                    rng = np.random.default_rng(mask_wid_rng)
                    print(f'\n\tCreating mask for mask_wid: {mask_wid} and rng: {mask_wid_rng} and compression: {hparams.compression} ...')
                    for name, param in sorted(model.named_parameters()):
                        indices = rng.choice(param.numel(), size=max(1, int(param.numel() // hparams.compression)), replace=False)
                        mask_indices[name] = torch.from_numpy(indices).long().cpu()
                    create_mask_end_time = time.time()
                    print(f'\t\tCreating mask completed in {create_mask_end_time - create_mask_start_time} seconds')
                    
                    # Prepare a validation dataset unknown to miners
                    print(f'Starting to generate pages for mask_wid: {mask_wid}, miner_uid: {miner_uid}')
                    generate_pages_start_time = time.time()
                    pages = random.sample(SubsetFineWebEdu2Loader.next_pages(
                        offset=mask_wid * hparams.pages_window_speed,
                        n_pages=10,
                        seed=miner_uid 
                    ), hparams.validator_pages_per_eval)
                    generate_pages_end_time = time.time()
                    print(f'Generated pages: {pages} in {generate_pages_end_time - generate_pages_start_time} seconds')

                    print(f'Starting to create dataset with batch_size: {config.actual_batch_size}, sequence_length: {hparams.sequence_length}')
                    create_dataset_start_time = time.time()
                    dataset = SubsetFineWebEdu2Loader(
                        batch_size=config.actual_batch_size,
                        sequence_length=hparams.sequence_length,
                        pages_info=pages,
                        tokenizer=hparams.tokenizer
                    )
                    create_dataset_end_time = time.time()
                    print(f'Dataset created with {dataset} batches in {create_dataset_end_time - create_dataset_start_time} seconds')

                    # Step 1: Zero the gradients of the model
                    print("Step 1: Zeroing the gradients of the model...")
                    step_start_time = time.time()
                    model.zero_grad()
                    step_end_time = time.time()
                    print(f"Step 1 completed in {step_end_time - step_start_time} seconds.")

                    # Step 2: Compute the gradient of the loss over the validation set
                    print("Step 2: Computing gradient over the validation set...")
                    step_start_time = time.time()
                    for idx, batch in enumerate(dataset):
                        input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                        labels = input_ids.clone()
                        labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                        with torch.amp.autocast(device_type=model.device.type, dtype=torch.bfloat16):
                            outputs = model(input_ids=input_ids, labels=labels)
                            loss = outputs.loss
                            loss.backward()  # Compute gradients
                    step_end_time = time.time()
                    print(f"Step 2 completed in {step_end_time - step_start_time} seconds.")

                    # Collect the gradients
                    print("Step 3: Collecting gradients...")
                    step_start_time = time.time()
                    gradients = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            gradients[name] = param.grad.detach().clone()
                        else:
                            # If the parameter did not receive a gradient, we set it to zero
                            gradients[name] = torch.zeros_like(param.data)
                    step_end_time = time.time()
                    print(f"Step 3 completed in {step_end_time - step_start_time} seconds.")

                    # Step 4: Flatten the gradients and the miner's update (mask values)
                    print("Step 4: Flattening gradients and miner's update...")
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
                    print(f"Step 4 completed in {step_end_time - step_start_time} seconds.")

                    # Step 5: Compute the dot product between the gradient vector and the miner's update vector
                    print("Step 5: Computing the dot product between the gradient and the miner's update...")
                    step_start_time = time.time()
                    dot_product = torch.dot(gradient_vector, update_vector)
                    step_end_time = time.time()
                    print(f'dot_product: {dot_product}')
                    print(f"Step 5 completed in {step_end_time - step_start_time} seconds.")
                    
                    # Optional regularization term
                    print("Step 6: Computing regularization term...")
                    step_start_time = time.time()
                    lambda_reg = hparams.update_norm_regularization  # Regularization coefficient; adjust as needed
                    update_norm = torch.norm(update_vector)
                    regularization = lambda_reg * update_norm.item()
                    step_end_time = time.time()
                    print(f'update_norm: {update_norm}')
                    print(f'regularization: {regularization}')
                    print(f"Step 6 completed in {step_end_time - step_start_time} seconds.")

                    # Step 7: Compute the reward
                    print("Step 7: Computing the reward...")
                    step_start_time = time.time()
                    reward = max(0.0, -dot_product.item() - regularization)
                    step_end_time = time.time()
                    print(f'reward: {reward}')
                    print(f"Step 7 completed in {step_end_time - step_start_time} seconds.")
                                        
                    # Set the weights
                    print("Step 8: Setting the weights...")
                    step_start_time = time.time()
                    weights[miner_uid] = (reward * hparams.weights_alpha) + ((1 - hparams.weights_alpha) * weights[miner_uid])
                    step_end_time = time.time()
                    print(f'weights[{miner_uid}]: {weights[miner_uid]}')
                    print(f"Step 8 completed in {step_end_time - step_start_time} seconds.")
                    
                    # Log the reward to wandb
                    if config.use_wandb:
                        wandb.log({f"R/{miner_uid}": reward, f"W/{miner_uid}": weights[miner_uid]})
                        step_end_time = time.time()

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
                                    
            # Every steps_per_master_upload steps we upload the master state of the model
            # This can be used for eval etc.
            if global_step % hparams.steps_per_master_upload == 1:
                # Upload a full copy of the model weights to master
                print('\nUploading master ...')
                start_time = time.time()
                model_state_dict = model.state_dict()
                upload_filename = f'master-{wallet.hotkey.ss58_address}.pt'
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
                print(f'Uploading master {upload_filename}@{config.bucket} completed in {time.time() - start_time} seconds.')
                  
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
    main(config)