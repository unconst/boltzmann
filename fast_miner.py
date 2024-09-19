

import os
import uuid
import boto3
import time
import tempfile
import bittensor as bt
from dotenv import dotenv_values
import torch
import numpy as np
import torch.optim as optim
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, LlamaConfig
from dataset import SubsetFineWebEdu2Loader

env_config = {**dotenv_values(".env"), **os.environ}  # Load environment variables.
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')  # AWS access key ID.
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')  # AWS secret access key.
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',  # AWS region.
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
subtensor = bt.subtensor('test')
metagraph = subtensor.metagraph(netuid=212)

step_size = 4
last_sync_block = (int(subtensor.block / step_size) * step_size) - 2 * step_size
upload_block_mask = (int(subtensor.block / step_size) * step_size) - 1 * step_size

print ('Get files...')
start_time = time.time()
buckets = []
meta_filenames = []
mask_filenames = []
for uid in metagraph.uids:
    buckets.append( subtensor.get_commitment(212, uid) )
    meta_filenames.append( f"mask-{str(metagraph.hotkeys[uid])}-{last_sync_block}_metadata.json")
    mask_filenames.append( f"mask-{str(metagraph.hotkeys[uid])}-{last_sync_block}.pt" )
print ( buckets)
print ( mask_filenames )
print(f'Get files completed in {time.time() - start_time} seconds')

# Create model.
print ('Create model....')
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained( 'gpt2', verbose=False, clean_up_tokenization_spaces=True )
tokenizer.pad_token = tokenizer.eos_token
model_config = LlamaConfig(
    vocab_size = tokenizer.vocab_size,
    hidden_size = 2040,
    num_hidden_layers = 12,
    num_attention_heads = 12,
    intermediate_size = 6144
)
model = LlamaForCausalLM( config = model_config )
print(f'Create model completed in {time.time() - start_time} seconds')

# Create Mask.
print ('Create download Mask.')
start_time = time.time()
mask_indices = {}
compression_factor = 300
np.random.seed( last_sync_block ) 
for name, param in model.named_parameters():
    next_mask = torch.from_numpy(np.random.rand(*param.shape) < (1 / compression_factor)).float()
    indices = next_mask.flatten().nonzero(as_tuple=False).flatten()
    mask_indices[ name ] = indices
print(f'Create downdload Mask completed in {time.time() - start_time} seconds')

# Get mask for the upload block.
print ('Create upload Mask.')
upload_block_mask = {}
compression_factor = 300
start_time = time.time()
np.random.seed( last_sync_block )  # Seed numpy's random generator with the upload block.
for name, param in model.named_parameters():
    upload_block_mask[name] = torch.from_numpy(np.random.rand(*param.shape) < (1 / compression_factor)).float()    
print(f'Create upload Mask completed in {time.time() - start_time} seconds')

# Download files using ThreadPoolExecutor
import concurrent.futures  # Add this import to avoid NameError

print('Download files.')
start_time = time.time()
n_downloaded = 0
temp_files = []

def download_file(bucket, filename):
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

print(f'Downloaded: {n_downloaded} in {time.time() - start_time} seconds')

# Loading state dicts
print ('Loading state dicts.')
start_time = time.time()
masks_dicts_values = {}
mask_count = 0
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
print(f'Applying {mask_count} masks')
start_time = time.time()
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
print(f'Applied state dicts in {time.time() - start_time} seconds')

# Delete files.
print ('Deleting files.')
start_time = time.time()
for file in temp_files:
    os.remove(file)
print(f'Deleting files completed in {time.time() - start_time} seconds')