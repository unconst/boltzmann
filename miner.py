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
import bittensor as bt
import concurrent.futures  
import torch.optim as optim
from typing import List, Tuple
from dotenv import dotenv_values
from transformers import LlamaForCausalLM 

from hparams import load_hparams
from dataset import SubsetFineWebEdu2Loader

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
    hparams = load_hparams()
    model = None
    upload_history = []  
    last_mask_sync = 0 
    last_master_sync = 0
    while True:
        try:    
            
            # Sync the current chain state and hparams.
            print ('Loading chain state:')
            start_time = time.time()
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
            print(f'Loading chain state completed in {time.time() - start_time} seconds') 
            
            # Sync the full model state every hparams.epoch_length
            print(f'Checking epoch sync:') 
            start_time = time.time() 
            if model == None or subtensor.block - last_master_sync > hparams.epoch_length:
                try:
                    master_uid = int(metagraph.S.argmax())
                    master_bucket = subtensor.get_commitment( config.netuid, master_uid )
                    master_hotkey = metagraph.hotkeys[ master_uid ]
                    master_filename = f'master-{master_hotkey}.pt'
                    unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                    CLIENT.download_file( master_bucket, master_filename, unique_temp_file )
                    master_state_dict = torch.load( unique_temp_file, map_location='cpu', weights_only = True )
                    model = LlamaForCausalLM( config = hparams.model_config ) 
                    model.load_state_dict( master_state_dict )
                    model.to(config.device)
                    model.train()
                    optimizer = optim.AdamW(
                        model.parameters(),
                        lr = config.learning_rate,  # Peak learning rate
                        betas = ( config.optimizer_beta1, config.optimizer_beta2 ), # B1 and B2
                        weight_decay = config.optimizer_weight_decay  # Weight decay
                    )
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams.epoch_length, gamma=0.1)
                    last_master_sync = subtensor.block 
                    last_mask_sync = last_master_sync
                except Exception as e:
                    print (f'No master. Waiting ...')
                    time.sleep(12)
                    continue
            print(f'Checking epoch sync: completed in {time.time() - start_time} seconds') 
            
            print(f'Getting block state:')
            start_time = time.time()  # Start timing
            block = subtensor.block
            all_sync_blocks = [ last_mask_sync + i + 1 for i in range( block - last_mask_sync )]
            last_mask_sync = block
            print(f'Getting block completed in {time.time() - start_time} seconds')
            
            # Get the mask for all sync blocks.
            print(f'Downloading masks for blocks: {all_sync_blocks}') 
            full_sync_start_time = time.time()
            for blk in all_sync_blocks:
                
                # Pull the filenames + buckets for all miners.
                print (f'Getting filenames for blk: {blk}...')
                start_time = time.time()
                if 'buckets' not in locals():
                    buckets = []
                    for uid in metagraph.uids:
                        buckets.append( subtensor.get_commitment(config.netuid, uid) )
                mask_filenames = []
                for uid in metagraph.uids:
                    mask_filenames.append( f"mask-{str(metagraph.hotkeys[uid])}-{blk}.pt" )
                print(f'Get filenames completed in {time.time() - start_time} seconds')
            
                # Download the masks from all valid files
                print(f'Downloading mask for blk: {blk}:')
                start_time = time.time()
                temp_files = []
                n_downloaded = 0
                def download_file( bucket, filename ):
                    try:
                        unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")
                        CLIENT.download_file(bucket, filename, unique_temp_file)
                        return unique_temp_file
                    except:
                        return None
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(download_file, bucket, filename) for bucket, filename in zip(buckets, mask_filenames)]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            temp_files.append(result)
                            n_downloaded += 1
                print(f'Downloading {n_downloaded} masks completed in {time.time() - start_time} seconds')
                
                # Break the loop when there is nothing to download.
                if n_downloaded == 0:
                    continue
                
                # Init the mask indicies using the block number.
                print(f'Creating sync mask for block: {blk}')
                mask_indices = {}
                torch.manual_seed( blk )
                start_time = time.time()
                for name, param in model.named_parameters():
                    param = param.to(config.device)
                    next_mask = (torch.rand(param.shape, device=config.device) < (1 / hparams.compression)).float()
                    indices = next_mask.flatten().nonzero(as_tuple=False).flatten()
                    mask_indices[name] = indices
                print(f'Creating sync block mask completed in {time.time() - start_time} seconds')
            
                # Load all masks as state dicts.
                print (f'Loading state dicts for block: {blk}:')
                start_time = time.time()
                mask_count = 0
                masks_dicts_values = {}
                for file in temp_files:
                    mask = torch.load( file, map_location='cpu', weights_only = True )
                    mask_count += 1
                    for name in mask.keys():
                        mask_values = mask[name]['values']
                        if torch.isnan(mask_values).any():
                            continue
                        param_shape = model.get_parameter(name).shape
                        indices = mask_indices[name] 
                        decompressed = torch.zeros(param_shape, device='cpu').flatten() 
                        decompressed[indices] = mask_values
                        if name not in masks_dicts_values:
                            masks_dicts_values[name] = decompressed.view(param_shape)
                        else:
                            masks_dicts_values[name] += decompressed.view(param_shape)
                print(f'Loading state dicts completed in {time.time() - start_time} seconds')
                
                # Average the masks before applying.
                print (f'Averaging {mask_count} masks for block: {blk}')
                start_time = time.time()
                for key in masks_dicts_values.keys():
                    masks_dicts_values[key] /= mask_count
                print(f'Averaged state dicts in {time.time() - start_time} seconds')
                
                # Set the average into the model.
                print(f'Applying {mask_count} masks for block: {blk}:')
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
                del masks_dicts_values
                print(f'Applying {mask_count} masks completed in {time.time() - start_time} seconds')
                
                # Delete files and clean up.
                print (f'Deleting files for block: {blk}.')
                start_time = time.time()
                for file in temp_files:
                    os.remove(file)
                print(f'Deleting files completed in {time.time() - start_time} seconds')
                
            # Print completion
            torch.cuda.empty_cache()
            print(f'Downloading masks for blocks: {all_sync_blocks} in {time.time() - full_sync_start_time} seconds')
                  
            # Select the block to produce a mask for.
            next_upload_block = subtensor.block - 1
            
            # Get the pages for this block and my_uid.
            # This is global and deterministic
            print ('Page loading:')
            start_time = time.time()  # Start timing
            pages = SubsetFineWebEdu2Loader.next_pages(
                offset = next_upload_block,
                n_pages = 1,
                seed = my_uid 
            )
            dataset = SubsetFineWebEdu2Loader(
                batch_size = config.batch_size,
                sequence_length = hparams.sequence_length,
                pages_info = pages,
                tokenizer = hparams.tokenizer
            )
            print(f'Page loading completed in {time.time() - start_time} seconds')
            
            # Train my model on the current page.
            print ('Training:')
            start_time = time.time()  # Start timing
            optimizer.zero_grad()
            avg_loss = 0
            n_batches = int( len(dataset.buffer) / (hparams.sequence_length * config.batch_size) )
            for idx, batch in enumerate( dataset ):
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                outputs = model(input_ids = input_ids, labels=labels)
                loss = outputs.loss / n_batches
                loss.backward()
                avg_loss += outputs.loss.item()
                del input_ids, labels, outputs
                torch.cuda.empty_cache()  
            optimizer.step()
            if config.use_wandb: wandb.log( { "step_loss": float( avg_loss / (idx+1) ) })
            print(f'Training completed in {time.time() - start_time} seconds')
            
            # Get the proper mask for my upload block + page.
            print(f'Creating upload mask:')
            start_time = time.time()  # Start timing
            upload_mask = {}
            torch.manual_seed(next_upload_block)  # Seed torch's random generator with the upload block.
            for name, param in model.named_parameters():
                param = param.to(config.device)
                next_mask = (torch.rand(param.shape, device=config.device) < (1 / hparams.compression)).float()
                upload_mask[name] = next_mask
            print(f'Creating upload block mask completed in {time.time() - start_time} seconds')
            
            # Mask the model values given the mask and produce a state dict.                
            print('Apply upload mask to model:')
            model_state_dict = model.state_dict()
            for name, param in model.named_parameters():
                param_mask = upload_mask[name].to(param.device)
                param_flat = param.flatten()
                mask_flat = param_mask.flatten()
                unmasked_indices = mask_flat.nonzero(as_tuple=False).flatten()
                unmasked_params = param_flat[unmasked_indices]
                model_state_dict[name] = {'values': unmasked_params}
            print(f'Applied mask to model completed in: {time.time() - start_time} seconds')

            # Upload the state dict of my masked weights.
            print('Uploading mask:')
            start_time = time.time()
            upload_filename = f'mask-{wallet.hotkey.ss58_address}-{next_upload_block}.pt'
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
            print(f'Uploading mask completed in {time.time() - start_time} seconds')

            # Delete old mask files and clean.
            print('Deleting history:')
            start_time = time.time()
            if len(upload_history) > 5:
                to_delete = upload_history.pop(0)
                CLIENT.delete_object(Bucket=config.bucket, Key=to_delete)
            print(f'Deleting history completed in {time.time() - start_time} seconds')
                 
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
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.95, help='Beta2 for the optimizer')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.1, help='Weight decay for the optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')    
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)    
    config = bt.config(parser)    
    config.subtensor.network = 'test'
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'    
    main(config)
