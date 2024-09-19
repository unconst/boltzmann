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

import wandb
import argparse
import traceback
import numpy as np
import bittensor as bt
import concurrent.futures  
import torch.optim as optim
from typing import List, Tuple
from types import SimpleNamespace
from transformers import Adafactor
from transformers import LlamaForCausalLM 

# Import constants and utility functions specific to the project.
from common import *
from dataset import SubsetFineWebEdu2Loader

def main(config):
    """
    Main function for the miner script.

    This function initializes the model, sets up training parameters, and
    enters the main training loop where the model is trained and periodically
    uploaded to the S3 bucket for validation.

    Args:
        config: The configuration object containing training parameters and settings.
    """
    # Print the configuration settings.
    print('\n', '-' * 40, 'Config', '-' * 40)
    print(config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)    
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')    
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)    
    print('\n', '-' * 40, 'Objects', '-' * 40)
    print(f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}')    
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
        
    # Init model
    hparams = load_hparams()
    upload_history = []    
    last_master_sync = 0
    while True:
        try:    
            
            print ('Loading chain state:')
            start_time = time.time()
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
            print(f'Loading chain state completed in {time.time() - start_time} seconds') 
            
            # Sync the full model state if we have gone further than the epoch.
            print(f'Checking epoch sync:')  # Print timing after this step
            start_time = time.time()  # Start timing
            if subtensor.block - last_master_sync > hparams.epoch_length:
                print ('Resyncing full training state.')
                try:
                    master_uid = int(metagraph.S.argmax())
                    master_meta = get_latest_metadata( key = 'model', uid = master_uid, metagraph = metagraph, subtensor = subtensor )
                    model = download_model( metadata = master_meta, device='cpu', CLIENT = CLIENT ) 
                    if model == None:
                        raise ValueError('No master.')
                    model.to(config.device)
                    model.train()
                    optimizer = optim.AdamW(
                        model.parameters(),
                        lr = config.learning_rate,  # Peak learning rate
                        betas = ( config.optimizer_beta1, config.optimizer_beta2 ), # B1 and B2
                        weight_decay = config.optimizer_weight_decay  # Weight decay
                    )
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams.epoch_length, gamma=0.1) 
                except Exception as e:
                    print (f'Error getting master: {e} Waiting ...')
                    time.sleep(12)
                    continue
                last_master_sync = subtensor.block
            print(f'Checking epoch sync: completed in {time.time() - start_time} seconds')  # Print timing after this step
            
            print(f'Getting block state:')
            start_time = time.time()  # Start timing
            block = subtensor.block
            step_size = hparams.blocks_per_step
            next_sync_block = (int(subtensor.block / step_size) * step_size) + step_size
            next_upload_block = (int(subtensor.block / step_size) * step_size)+ (step_size * 2)
            print(f'Getting block completed in {time.time() - start_time} seconds')  # Print timing after this step
            
            print ('Getting filenames...')
            start_time = time.time()
            if 'buckets' not in locals():
                buckets = []
                for uid in metagraph.uids:
                    buckets.append( subtensor.get_commitment(config.netuid, uid) )
            mask_filenames = []
            for uid in metagraph.uids:
                mask_filenames.append( f"mask-{str(metagraph.hotkeys[uid])}-{next_sync_block}.pt" )
            print(f'Get filenames completed in {time.time() - start_time} seconds')
            
            # Get the mask for the sync block.
            print(f'Creating sync mask:')
            start_time = time.time()  # Start timing
            mask_indices = {}
            np.random.seed( next_sync_block ) 
            for name, param in model.named_parameters():
                next_mask = torch.from_numpy(np.random.rand(*param.shape) < (1 / hparams.compression)).float()
                indices = next_mask.flatten().nonzero(as_tuple=False).flatten()
                mask_indices[ name ] = indices
            print(f'Creating sync block mask completed in {time.time() - start_time} seconds')  # Print timing after this step
                
            # Get mask for the upload block.
            print(f'Creating upload mask:')
            start_time = time.time()  # Start timing
            upload_block_mask = {}
            np.random.seed( next_upload_block )  # Seed numpy's random generator with the upload block.
            for name, param in model.named_parameters():
                upload_block_mask[name] = torch.from_numpy(np.random.rand(*param.shape) < (1 / hparams.compression)).float()
            print(f'Creating upload block mask completed in {time.time() - start_time} seconds')  # Print timing after this step
                
            # Wait until uploads.
            print ('Waiting for sync:')
            start_time = time.time()  # Start timing
            while True:
                block = subtensor.block
                if block >= next_sync_block:
                    break
                print (f'Waiting for sync block: {next_sync_block} current: {block}')
                time.sleep(4)
                continue
            print(f'Waiting for sync block completed in {time.time() - start_time} seconds')  # Print timing after this step
            
            print('Downloading masks:')
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
            
            # Loading state dicts
            print ('Loading state dicts:')
            start_time = time.time()
            mask_count = 0
            masks_dicts_values = {}
            for file in temp_files:
                mask = torch.load( file, map_location='cpu', weights_only = True )
                for name in mask.keys():
                    param_shape = model.get_parameter(name).shape
                    mask_values = mask[name]['values']
                    indices = mask_indices[name] 
                    decompressed = torch.zeros(param_shape, device='cpu').flatten() 
                    decompressed[indices] = mask_values
                    if name not in masks_dicts_values:
                        masks_dicts_values[name] = decompressed.view(param_shape)
                    else:
                        masks_dicts_values[name] += decompressed.view(param_shape)
            print(f'Loading state dicts completed in {time.time() - start_time} seconds')
            
            # Average the mask values
            print (f'Averaging {mask_count} masks')
            start_time = time.time()
            for key in masks_dicts_values.keys():
                masks_dicts_values[key] /= mask_count
            print(f'Averaged state dicts in {time.time() - start_time} seconds')
            
            # Set these values into the model
            print(f'Applying {mask_count} masks:')
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
            torch.cuda.empty_cache()
            print(f'Applying {mask_count} masks completed in {time.time() - start_time} seconds')  # Print timing after this step
            
            # Delete files.
            print ('Deleting files.')
            start_time = time.time()
            for file in temp_files:
                os.remove(file)
            print(f'Deleting files completed in {time.time() - start_time} seconds')
                                                    
            # Get current block page for miner.
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
            print(f'Page loading completed in {time.time() - start_time} seconds')  # Print timing after this step
            
            # Train model on page.
            print ('Training:')
            start_time = time.time()  # Start timing
            optimizer.zero_grad()
            avg_loss = 0
            n_batches = int( len(dataset.buffer) / (hparams.sequence_length * config.batch_size) )
            for idx, batch in enumerate( dataset ):
                # Break the training if we are past the training block.
                block = subtensor.block
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                outputs = model(input_ids = input_ids, labels=labels)
                loss = outputs.loss / n_batches
                loss.backward()
                avg_loss += outputs.loss.item()
                if config.use_wandb: wandb.log( { "training_loss": float(outputs.loss.item()) })
                print ( 'batch', f'{idx}/{n_batches}', 'block', block, 'Loss', outputs.loss.item() )
                if block >= next_upload_block - 2:
                    optimizer.step()
                    print (f'Break training on {block} with next upload: {next_upload_block}')
                    break
                else:
                    del input_ids, labels, outputs
                    torch.cuda.empty_cache()  
            # dont even upload we didnt train.
            if config.use_wandb: wandb.log( { "step_loss": float( avg_loss / (idx+1) ) })
            print(f'Training completed in {time.time() - start_time} seconds')  # Print timing after this step
                
            # Upload the masked weights.
            print ('Uploading mask:')
            start_time = time.time()
            upload_history.append( upload_model(
                key = 'mask',
                wallet = wallet,
                model = model,
                block = next_upload_block, # Key the mask with the upload block here.
                extras = {},  # Additional metadata can be added here.
                bucket = config.bucket,
                CLIENT = CLIENT,
                mask = upload_block_mask,
                with_indicies = False,
            ))
            print(f'Uploading mask completed in {time.time() - start_time} seconds')
            
            print(f'Deleting history:')
            start_time = time.time()
            if len(upload_history) > 10: # should be full epoch.
                to_delete = upload_history.pop(0)
                CLIENT.delete_object( Bucket=config.bucket, Key=to_delete.filename )
                CLIENT.delete_object( Bucket=config.bucket, Key=to_delete.metadata_filename )
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
    # Create an argument parser for command-line options.
    parser = argparse.ArgumentParser(description='Miner script')
    
    # Add command-line arguments with default values and help descriptions.
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
