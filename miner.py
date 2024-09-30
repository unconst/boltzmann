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
import wandb
import torch
import argparse
from tqdm import tqdm
import bittensor as bt
from typing import List
import torch.optim as optim
from dotenv import dotenv_values
from transformers import LlamaForCausalLM 
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import local files.
from common import *
from hparams import load_hparams
from dataset import SubsetFineWebEdu2Loader

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Main function.
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
        # Delete all runs with my name and create a new one.
        try: [run.delete() for run in wandb.Api().runs(path=config.project) if run.name == f'M{my_uid}' and print(f'Deleting old run: {run}')]
        except: pass
        wandb.init(project=config.project, resume='allow', name=f'M{my_uid}', config=config)
        
    # Init training state.
    print('\n', '-' * 40, 'Hparams', '-' * 40)
    hparams = load_hparams()
    model = LlamaForCausalLM( config = hparams.model_config )
    model.to(config.device)
    model.train()
    optimizer = optim.AdamW(
        model.parameters(),
        lr = hparams.learning_rate,  # Peak learning rate
        betas = ( hparams.optimizer_beta1, hparams.optimizer_beta2 ), # B1 and B2
        weight_decay = hparams.optimizer_weight_decay,  # Weight decay
        foreach = True,  # more memory usage, but faster
    )
    scheduler = CosineAnnealingLR( optimizer, T_max = hparams.cosine_epoch_length, eta_min=hparams.eta_min, last_epoch=-1 )
        
    # Load bucket information.
    buckets = []
    for uid in tqdm(metagraph.uids):
        try: buckets.append('decis')
        except: buckets.append(None)
        
    # Returns the mask window based on a block.
    def block_to_mask( block: int ) -> int:
        return int(block / hparams.mask_window_length)

    # Updates chain state on a background loop.
    async def update(stop_event):
        print (f'[update]: Start update loop.')
        # Forever until the stop event is set.
        while not stop_event.is_set():
            hparams = load_hparams()
            subtensor = bt.subtensor(config=config)
            metagraph = subtensor.metagraph(netuid=config.netuid)
            if len(buckets) != len(metagraph.uids):
                for uid in tqdm(metagraph.uids):
                    # try: buckets.append(subtensor.get_commitment(config.netuid, uid))
                    try: buckets.append('decis')
                    except: buckets.append(None)
            print (f'[update]: Updated state. Waiting {60} seconds.')
            await asyncio.sleep(60)  # Add a sleep to prevent tight loop

    # Pulls masks from other peers on a background loop.  
    async def download(stop_event):
        print (f'[download]: Start download loop.')
        # Forever until the stop event is set.
        while not stop_event.is_set(): 
            current_mask = block_to_mask(subtensor.block)
            # Download masks from current and previous windows.
            masks = [current_mask - 2, current_mask - 1, current_mask ]
            print (f'[download]: downloading masks for: {masks} ... ')
            files_for_masks = await download_files_for_buckets_and_masks( buckets = buckets, masks = masks )
            print (f'[download]: downloaded {sum([len(files_for_masks[k]) for k in files_for_masks.keys()])} files for masks: {masks}')
            print (f'[download]: waiting {2}s ... ')
            await asyncio.sleep(2)
       
    # Apply masks to model.
    async def apply(stop_event):
        print (f'[apply]: Start apply loop.')
        # Forever until the stop event is set.
        while not stop_event.is_set():
            # Wait until the mask has changed.
            current_mask = block_to_mask( subtensor.block )
            indices = await get_indicies_for_mask( model, current_mask, hparams.compression )
            while block_to_mask(subtensor.block) == current_mask:
                print (f'[apply]: waiting {1}s ... ')
                await asyncio.sleep(1)
            print (f'[apply]: Applying masks for: {current_mask - 1} ')

            # Load masks from previous window.
            mask_files = await get_files_for_mask_from_temp( mask = current_mask - 1 )
            print (f'[apply]: Loaded {len(mask_files)} masks for: {current_mask - 1} ')

            # Apply masks to model and average them.
            masks_per_param = { name: 0 for name in model.named_parameters() }
            for mask_info in mask_files:
                try:
                    mask = torch.load(mask_info.temp_file, map_location=torch.device(model.device), weights_only=True)
                    for name, param in model.named_parameters():
                        if name not in indices: continue
                        if name not in mask: continue
                        values = mask[name].to(model.device)
                        indices = indices[name].to(model.device)
                        param.data.view(-1)[indices] += values 
                        masks_per_param[name] += 1
                        del values
                except Exception as e:
                    pass
            for name, param in model.named_parameters():
                if name not in masks_per_param: continue
                if name not in indices: continue
                if masks_per_param[name] == 0: continue # Nothing to average.
                indices = indices[name].to(config.device)
                param.data.view(-1)[indices] /= (masks_per_param[name] + 1)
            print (f'[apply]: Applied {len(mask_files)} masks for: {current_mask - 1}')
                
    # Trains for model.  
    async def train(stop_event):
        print (f'[train]: Start train loop.')
        # Forever until the stop event is set.
        while not stop_event.is_set():
            # Get current mask
            current_mask = block_to_mask(subtensor.block)
            # Train for the current mask.
            print (f'[train]: Loading dataset for mask: {current_mask}')
            pages = SubsetFineWebEdu2Loader.next_pages(
                offset = subtensor.block * hparams.pages_window_speed,
                n_pages = 10,
                seed = my_uid 
            )
            dataset = SubsetFineWebEdu2Loader(
                batch_size = config.actual_batch_size,
                sequence_length = hparams.sequence_length,
                pages_info = pages,
                tokenizer = hparams.tokenizer
            )
            print (f'[train]: Loaded dataset for mask: {current_mask}')

            # Train my model on the current page.
            print (f'[train]: Training on dataset for mask: {current_mask}')
            torch.cuda.empty_cache() # Empty cache going into the training step.
            optimizer.zero_grad() # Clear any lingering grads.
            total_loss = 0.0
            total_steps = hparams.desired_batch_size // config.actual_batch_size
            for idx, batch in enumerate( dataset ):
                input_ids = torch.tensor(batch, dtype=torch.long).to(model.device)
                labels = input_ids.clone()
                labels = torch.where(labels == hparams.tokenizer.pad_token_id, -100, labels)
                with torch.amp.autocast( device_type = model.device.type, dtype = torch.bfloat16 ):  # Enable autocasting for mixed precision
                    outputs = model(input_ids = input_ids, labels=labels)
                total_loss += outputs.loss.item()
                loss = outputs.loss / (total_steps + 1) # Divide by number of accumulations.
                loss.backward()
                # Break on total steps of if we have 1 block left.
                if idx >= total_steps - 1 or block_to_mask( subtensor.block ) != current_mask:
                    break
            print (f'[train]: Trained {total_steps} steps with an average loss of {total_loss}.')
                
            # Apply step and clean memory.
            print (f'[train]: Stepping for mask: {current_mask}')
            if hparams.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
            optimizer.step()
            scheduler.step()  # Update the learning rate.
            optimizer.zero_grad()                        
            del input_ids, labels, outputs
            torch.cuda.empty_cache()
            print (f'[train]: Finished step for mask: {current_mask}')

            # Upload our model mask to S3.
            print (f'[train]: Uploading for mask: {current_mask}')
            await upload_mask( config.bucket, model, current_mask, wallet, hparams.compression )
            print (f'[train]: Finished upload for mask: {current_mask}')

    # Start background threads.
    stop_event = asyncio.Event()
    tasks = [
        update(stop_event),
        download(stop_event),
        apply(stop_event),
        train(stop_event)
    ]
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        stop_event.set()
        await asyncio.gather(*tasks)

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
    asyncio.run(main(config))
