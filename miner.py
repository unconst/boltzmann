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
    
    # Initialize Bittensor wallet with the provided configuration.
    wallet = bt.wallet(config=config)
    # Initialize Bittensor subtensor with the provided configuration.
    subtensor = bt.subtensor(config=config)
    # Retrieve the metagraph for the specified netuid.
    metagraph = subtensor.metagraph(netuid=config.netuid)
    
    # Ensure the wallet's hotkey is registered on the subnet.
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    
    # Get the UID (unique identifier) of the wallet's hotkey in the metagraph.
    my_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    
    # Print initialized objects for debugging purposes.
    print('\n', '-' * 40, 'Objects', '-' * 40)
    print(f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}')
    
    # Check if the bucket committed on-chain matches the configured bucket.
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
            
            # Load chain state.
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
            
            # Sync the full model state if we have gone further than the epoch.
            if subtensor.block - last_master_sync > 100:
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
                except Exception as e:
                    print (f'Error getting master: {e} Waiting ...')
                    time.sleep(12)
                    continue
                last_master_sync = subtensor.block
            
            # Get block.
            block = subtensor.block
            step_size = hparams.blocks_per_step
            next_sync_block = (int(subtensor.block / step_size) * step_size) + step_size
            next_upload_block = (int(subtensor.block / step_size) * step_size)+ (step_size * 2)
            while True:
                block = subtensor.block
                if block >= next_sync_block:
                    break
                print (f'Waiting for sync block: {next_sync_block} current: {block}')
                time.sleep(4)
                continue
            
            def sync_state( sync_block: int ):
                # Get the mask for the sync block.
                mask_indices = {}
                compression_factor = hparams.compression
                print(f'Creating {compression_factor}X compression mask for block: {sync_block}')
                # We seed the mask from the block height.
                np.random.seed( sync_block ) 
                for name, param in model.named_parameters():
                    next_mask = torch.from_numpy(np.random.rand(*param.shape) < (1 / compression_factor)).float()
                    indices = next_mask.flatten().nonzero(as_tuple=False).flatten()
                    mask_indices[ name ] = indices
                
                # Sync and average all the masks from peers on the sync block.
                masks_dicts_values = {}
                mask_count = 0
                for uid in metagraph.uids:
                    metadata = get_metadata_for_block( 
                        key = 'mask', 
                        uid = uid, 
                        block = sync_block,
                        metagraph = metagraph, 
                        subtensor = subtensor,
                    )
                    if metadata == None: continue
                    # Download the compressed state_dict.
                    mask = download_model( metadata = metadata, device='cpu', CLIENT=CLIENT, state_dict = True )
                    if mask == None: continue
                    mask_count += 1
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
                print (f'Pulled {mask_count} masks')

                # Average the mask values
                print (f'Averaging {mask_count} masks')
                for key in masks_dicts_values.keys():
                    masks_dicts_values[key] /= mask_count
                    # TODO: Check for division by zero in case mask_count is zero
                    # TODO: Ensure that masks_dicts_values[key] is not None before performing division
                    
                # Set these values into the model
                print (f'Applying {mask_count} masks')
                for name, param in model.named_parameters():
                    indices = mask_indices[name]
                    if name in masks_dicts_values:
                        if masks_dicts_values[name].shape == param.shape:
                            # Overload the indicies from the mask.
                            on_device = masks_dicts_values[name].to(model.device)
                            param.data[indices] = on_device[indices]
                            del on_device
                        else:
                            print(f"Shape mismatch for {name}: expected {param.shape}, got {masks_dicts_values[name].shape}")
                del masks_dicts_values
                torch.cuda.empty_cache()
                            
            # Sync the state from all peers.
            sync_state(next_sync_block)
            print (f'Synced state by {subtensor.block} with upload in {next_upload_block}')
            
            # Get current block page for miner.
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
            
            # Train model on page.
            for idx, batch in enumerate( dataset ):
                # Break the training if we are past the training block.
                block = subtensor.block
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                outputs = model(input_ids = input_ids, labels=labels)
                outputs.loss.backward()
                optimizer.step()
                if config.use_wandb: wandb.log( { "loss": outputs.loss.item(), f'Incentive{my_uid}': float(metagraph.I[ my_uid ]) })
                print ( 'block', block, 'Loss', outputs.loss.item() )
                del input_ids, labels, outputs
                torch.cuda.empty_cache()  
                if block >= next_upload_block - 1:
                    print (f'Break training on {block} with next upload: {next_upload_block}')
                    break

            # Get mask for the upload block.
            upload_block_mask = {}
            compression_factor = hparams.compression
            print(f'Creating {compression_factor}X compression mask for block: {next_upload_block}')
            np.random.seed( next_upload_block )  # Seed numpy's random generator with the upload block.
            for name, param in model.named_parameters():
                upload_block_mask[name] = torch.from_numpy(np.random.rand(*param.shape) < (1 / compression_factor)).float()
                
            # Upload the masked weights.
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
            
            # Delete history over allowed.
            if len(upload_history) > 10: # should be full epoch.
                to_delete = upload_history.pop(0)
                CLIENT.delete_object( Bucket=config.bucket, Key=to_delete.filename )
                CLIENT.delete_object( Bucket=config.bucket, Key=to_delete.metadata_filename )
                 
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
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
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
