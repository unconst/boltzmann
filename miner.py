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
import bittensor as bt
import numpy as np
from types import SimpleNamespace
from transformers import Adafactor
from typing import List, Tuple
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
    model = LlamaForCausalLM( config = hparams.model_config )    
    optimizer = Adafactor(
        model.parameters(),
        lr = config.learning_rate,
        relative_step = False
    )
    model.to(config.device)
    model.train()
    master = None
    upload_history = []
    total_pages = 0
    total_epochs = 0
    total_failed_sync = 0
    total_success_sync = 0
    while True:
        try:    
            # Load chain state.
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)

            # Get master.
            print ('Get master meta.')
            master_uid = int(metagraph.S.argmax())
            master_meta = get_latest_metadata( key = 'model', uid = master_uid, metagraph = metagraph, subtensor = subtensor, CLIENT = CLIENT)
            if master_meta == None:
                print ('Waiting for master...')
                time.sleep(12)
                continue
            
            # Initial pass, download the master directly.
            print ('Getting master.')
            if master == None or not hasattr( master_meta, 'deltas' ):
                master = download_model( metadata = master_meta, device='cpu', CLIENT = CLIENT ) 
                print ('Downloaded the master.')
                
            # If master_meta is not my state, sync.
            if hash_model( master ) != master_meta.model_hash:
                # Merge all miner deltas.
                delta_metas = [ SimpleNamespace( **d ) for d in master_meta.deltas]
                total_pages = sum([ d.n_pages for d in delta_metas])
                for meta in delta_metas:
                    try:
                        delta = download_model( metadata = meta, device='cpu', CLIENT=CLIENT )
                        for (name, master_param), (_, delta_param) in zip( master.named_parameters(), delta.named_parameters() ):
                            master_param.data.add_( (meta.n_pages/total_pages) * delta_param.data.to( master.device ) )
                        print ('Applied delta.')
                    except Exception as e:
                        print (f'Failed to apply deltas with error: {e}')
                        break
                print (f'Applied deltas with {total_pages} applied')
                
            # Checking if the previous sync merged properly.
            if hash_model( master ) != master_meta.model_hash:
                # Merge was unsuccessful, downloading full.
                print ('Failed to sync master from deltas, downloading full state...')
                master = download_model( metadata = master_meta, device='cpu', CLIENT = CLIENT ) 
                total_failed_sync += 1
            else:
                # Merge was successful.
                print ('Successfully syncd master state from deltas.')
                total_success_sync += 1
            if config.use_wandb: wandb.log({ "total_failed_sync": total_failed_sync, "total_success_sync": total_success_sync })
            
            # Check for failed state sync.
            if master == None:
                print ('Master was None, continue...')
                continue
            
            # Copy the master state into the model.
            print ('Sink delta into the master.')
            for (name, model_param), (_, master_param) in zip(model.named_parameters(), master.named_parameters()):
                model_param.data.copy_(master_param.data.to(model.device))
            
            # Build the current mask.
            mask = {}
            compression_factor = hparams.compression
            print (f'Creating Mask with compression: {compression_factor}')
            for name, param in model.named_parameters():
                mask[name] = (torch.rand_like(param) < (1 / compression_factor)).float() 
                
            # Epochs start here. 
            total_epochs += 1
            if config.use_wandb: wandb.log({ "total_epochs": total_epochs })
            
            # Get next page.
            print ('Training until next master.')
            n_pages = 0
            while True:
                
                # Break on state change.
                next_master_meta = get_latest_metadata( key = 'model', uid = master_uid, metagraph = metagraph, subtensor = subtensor, CLIENT = CLIENT)
                if next_master_meta == None or next_master_meta.model_hash != master_meta.model_hash:
                    break
                
                print ('Get next dataset...')
                n_pages += 1
                total_pages += 1
                if config.use_wandb: wandb.log({ "n_pages": n_pages, "total_pages": total_pages })
                pages = SubsetFineWebEdu2Loader.next_pages(
                    offset = subtensor.block * hparams.window_speed,
                    n_pages = 1,
                    seed = my_uid 
                )
                dataset = SubsetFineWebEdu2Loader(
                    batch_size = config.actual_batch_size,
                    sequence_length = hparams.sequence_length,
                    pages_info = pages,
                    tokenizer = hparams.tokenizer
                )
                # Train the model with the mask.
                print ('Start training...')
                # dont_upload = False
                for idx, batch in enumerate( dataset ):
                    input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                    labels = input_ids.clone()
                    labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                    outputs = model(input_ids = input_ids, labels=labels)
                    outputs.loss.backward()
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            # Mask the gradient.
                            param.grad.mul_( mask[ name ].to( param.grad.device ).to(param.grad.dtype) )  
                    optimizer.step()
                    if config.use_wandb: wandb.log({ "loss": outputs.loss.item(), f'Incentive{my_uid}': metagraph.I[ my_uid ] })
                    print ( 'Loss', outputs.loss.item() )
                    del input_ids, labels, outputs
                    torch.cuda.empty_cache()
                
                # Compute the delta between the model and the master.
                print ('Get Delta.')
                delta = copy.deepcopy( model ).to('cpu')
                for (name, delta_param), (_, master_param) in zip(delta.named_parameters(), master.named_parameters()):
                    delta_param.data.sub_(master_param.data.to('cpu'))   
                    
                print ('Upload Delta.')
                upload_history.append( upload_model(
                    key = 'delta',
                    wallet = wallet,
                    model = delta,
                    block = int(time.time()),  # Use current timestamp as block number.
                    extras = {'n_pages': n_pages, 'master_hash': master_meta.model_hash },  # Additional metadata can be added here.
                    bucket = config.bucket,
                    CLIENT = CLIENT,
                    mask = mask,
                ))
                # Delete history over allowed.
                if len(upload_history) > 3:
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
    parser.add_argument('--desired_batch_size', type=int, default=1, help='Desired total batch size for training')
    parser.add_argument('--actual_batch_size', type=int, default=1, help='Actual batch size per step')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.95, help='Beta2 for the optimizer')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.1, help='Weight decay for the optimizer')
    parser.add_argument('--pages_per_epoch', type=int, default=1, help='Number of pages to train per epoch')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')    
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)    
    config = bt.config(parser)    
    config.subtensor.network = 'test'
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'    
    main(config)
