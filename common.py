
# The MIT License (MIT)
# Copyright © 2024 Chakana.tech

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
import math
import time
import json
import torch
import hashlib
import tempfile
import bittensor as bt
from types import SimpleNamespace
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from typing import Dict, List, Optional, Tuple, Any

def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024.0

def get_latest_metadata_block( hotkey:str, bucket:str, CLIENT ) -> int:
    """
    Retrieves the latest metadata block number for a given hotkey from the specified bucket.

    This function lists all objects in the specified bucket, filters the filenames that match the pattern
    'model-{hotkey}-<block>.pt', and extracts the block number from the filenames. It returns the highest
    block number found.

    Args:
        hotkey (str): The hotkey associated with the model files.
        bucket (str): The name of the bucket where the model files are stored.
        CLIENT: The client used to interact with the storage service.

    Returns:
        int: The highest block number found for the given hotkey. Returns -1 if no matching files are found.
    """
    response = CLIENT.list_objects_v2( Bucket = bucket )
    # TODO( const ): needs pagination.
    file_names = [content['Key'] for content in response.get('Contents', [])]
    max_block = -1
    latest_file = None
    for file_name in file_names:
        if file_name.startswith(f'model-{hotkey}-') and file_name.endswith('.pt'):
            try:
                block = int(file_name.split('-')[-1].split('.')[0])
                if block > max_block:
                    max_block = block
                    latest_file = file_name
            except ValueError:
                continue
    return max_block

def hash_model( module: torch.nn.Module ) -> str:
    """
    Generates a SHA-256 hash of the model's state dictionary.

    This function iterates through the model's state dictionary, concatenates the byte representation
    of each parameter, and then generates a SHA-256 hash of this concatenated byte string.

    Args:
        model (torch.nn.Module): The model to hash.

    Returns:
        str: The SHA-256 hash of the model's state dictionary.
    """
    if module == None:
        return "0x0000000000000000000000000000000000000000000000000000000000000000"
    
    # Extract the state dictionary from the module which contains all the parameters.
    module_state_dict = module.state_dict()
    
    # Concatenate all the model state values into a single byte string.
    concatenated_model_states_bytes = b''.join(
        [value.cpu().numpy().tobytes() for value in module_state_dict.values()]
    )
    
    # Generate a SHA-256 hash from the concatenated bytes.
    module_hash = hashlib.sha256(concatenated_model_states_bytes).hexdigest()
    return module_hash

def get_latest_metadata_for_hotkey_and_bucket(
        hotkey: str,
        bucket: str,
        CLIENT
    ) -> SimpleNamespace:
    # Latest block
    latest_block = get_latest_metadata_block( hotkey, bucket, CLIENT )
    if latest_block == -1:
        # Metadata does not exist.
        return None
    
    # Define the filenames for the model and its metadata
    filename = f"model-{ hotkey }-{latest_block}.pt"
    metadata_filename = f"model-{ hotkey }-{latest_block}_metadata.json"        
    
    # Get the metadata of the model file from the storage service
    response = CLIENT.head_object(Bucket = bucket, Key=filename)
    
    # Extract and calculate metadata information
    metadata = {}
    metadata['last_modified'] = int(response['LastModified'].timestamp())
    metadata['blocks_since_modified'] = int((time.time() - int(response['LastModified'].timestamp())) / 12)
    metadata['size'] = response['ContentLength']
    metadata['bucket'] = bucket
    metadata['filename'] = filename
    metadata['metadata_filename'] = metadata_filename

    # Get the metadata file from the storage service
    metadata_response = CLIENT.get_object(Bucket=bucket, Key=metadata_filename)
    
    # Read and update the metadata with the content of the metadata file
    metadata_json = json.loads(metadata_response['Body'].read().decode('utf-8'))
    metadata.update(metadata_json)
    return SimpleNamespace(**metadata)

def get_latest_metadata( 
        uid: int, 
        metagraph, 
        subtensor, 
        CLIENT 
    ) -> Optional[ SimpleNamespace ]:
    """
    Retrieves metadata for a specified model from a storage service.

    Args:
        uid (int): The unique identifier for the model.
        block (int): The block value where this file can be found.
        metagraph: The bittensor metagraph containing network information.
        subtensor: The bittensor subtensor object used to interact with the network.
        CLIENT: The client used to interact with the storage service.

    Returns:
        Optional[SimpleNamespace]: A namespace containing the metadata if successful, otherwise None.
    """
    try:
        # Get the bucket name using the subtensor and metagraph information
        bucket = subtensor.get_commitment(metagraph.netuid, uid)
        hotkey = metagraph.hotkeys[uid]
        metadata = get_latest_metadata_for_hotkey_and_bucket( hotkey = hotkey, bucket = bucket, CLIENT = CLIENT )
        if metadata != None:
            metadata.uid = int(uid)
            return metadata
        else: 
            return None
    except Exception as e:
        print ( e )
        # Return None if any exception occurs
        return None
    

def upload_model( 
        wallet: 'bt.wallet',
        model: torch.nn.Module, 
        block: int,
        extras: Dict[ str, object ],
        bucket: str,
        CLIENT,
    ) -> SimpleNamespace:
    """
    Uploads a model to a specified bucket along with its metadata.

    Args:
        wallet (bt.wallet): The wallet containing the hotkey used to generate the filename.
        model (torch.nn.Module): The model to be uploaded.
        extras (Dict[str, object]): Additional metadata to be uploaded with the model.
        bucket (str): The bucket to upload the model to.
        CLIENT: The client used to interact with the storage service.

    Returns:
        None
    """
    start_time = time.time()  # Record the start time for the upload process
    model_state_dict = model.state_dict()  # Get the state dictionary of the model

    # Extract the configuration from the model and update extras with model type and configuration
    if isinstance(model, LlamaForCausalLM):
        config = model.config  # Get the configuration of the Llama model
        extras.update({
            'model_type': 'llama',  # Add model type to extras
            'model_config': config.to_dict()  # Add model configuration to extras
        })
    elif isinstance(model, GPT2LMHeadModel):
        config = model.config  # Get the configuration of the GPT-2 model
        extras.update({
            'model_type': 'gpt2',  # Add model type to extras
            'model_config': config.to_dict()  # Add model configuration to extras
        })
    # Add model hashes.
    extras['model_hash'] = hash_model( model )

    # Generate filenames for the model and its metadata
    filename = f'model-{wallet.hotkey.ss58_address}-{block}.pt'  # Filename for the model
    metadata_filename = f"model-{wallet.hotkey.ss58_address}-{block}_metadata.json"  # Filename for the metadata

    # Upload the metadata to the storage service
    metadata_buffer = io.BytesIO(json.dumps(extras).encode('utf-8'))  # Create a buffer for the metadata
    CLIENT.upload_fileobj(metadata_buffer, bucket, metadata_filename)  # Upload the metadata buffer to the storage service

    # Grant read and list permissions to all users for the metadata
    CLIENT.put_object_acl(
        Bucket=bucket,
        Key=metadata_filename,
        GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
        GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
    )
    
    # Upload the model to the storage service
    start_time = time.time()  # Record the start time for the upload process.
    with io.BytesIO() as module_buffer:
        torch.save(model_state_dict, module_buffer)  # Save the model state dictionary to the buffer
        module_buffer.seek(0)  # Reset the buffer's position to the beginning
        CLIENT.upload_fileobj(module_buffer, bucket, filename)  # Upload the model buffer to the storage service

    # Grant read and list permissions to all users for the model
    CLIENT.put_object_acl(
        Bucket=bucket,
        Key=filename,
        GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
        GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
    )

    # Log the completion of the upload process with the time taken
    returned_metadata = get_latest_metadata_for_hotkey_and_bucket( hotkey = wallet.hotkey.ss58_address, bucket = bucket, CLIENT = CLIENT )
    print(f"Uploaded model to {filename}@{bucket} of size: {human_readable_size(returned_metadata.size)} in: {time.time() - start_time} seconds.")
    return returned_metadata

def download_model( 
        metadata: SimpleNamespace, 
        device: str, 
        CLIENT,
    ) -> Optional[ torch.nn.Module ]:
    """
    Downloads a model from a specified bucket and loads it onto the specified device.

    Args:
        metadata (SimpleNamespace): Metadata containing information about the model to be downloaded.
        device (str): The device to load the model onto (e.g., 'cpu' or 'cuda').
        CLIENT: The client used to interact with the storage service.

    Returns:
        Optional[torch.nn.Module]: The downloaded model if successful, otherwise None.
    """
    try:
        print(f'Downloading model from {metadata.filename}@{metadata.bucket}')  # Log the start of the download process
        start_time = time.time()  # Record the start time for the download

        # Check the model type and initialize the appropriate model configuration and model
        if metadata.model_type == "llama":
            model_config = LlamaConfig(**metadata.model_config)  # Create Llama model configuration
            model = LlamaForCausalLM(model_config)  # Initialize Llama model
        if metadata.model_type == "gpt2":
            model_config = GPT2Config(**metadata.model_config)  # Create GPT-2 model configuration
            model = GPT2LMHeadModel(model_config)  # Initialize GPT-2 model
            
        # Generate a unique temporary file path using uuid
        unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")  # Create a unique temp file name

        # Download the model file from the storage service
        CLIENT.download_file(metadata.bucket, metadata.filename, unique_temp_file)  # Download the model file to a unique file path

        # Load the model state dict from the unique temporary file
        new_model_state_dict = torch.load(unique_temp_file, map_location=torch.device(device), weights_only=True)  # Load the model state dict
                
        model.load_state_dict(new_model_state_dict)  # Load the state dict into the model
        model.to(device)  # Move the model to the specified device

        # Remove the temporary file after use
        os.remove(unique_temp_file)  # Delete the unique temporary file

        # Log the completion of the download process with the time taken
        print(f"Downloaded model from {metadata.filename}@{metadata.bucket} of size: {human_readable_size(metadata.size)} in: in {time.time() - start_time} seconds.")
        return model  # Return the downloaded model
    except Exception as e:
        print (f'Error while downloading model from {metadata.filename}@{metadata.bucket} with error {e}.')
        return None
    
def save_history(
    wallet: 'bt.wallet',
    history: Dict[str, Any],
    bucket: str,
    CLIENT,
) -> None:
    """
    Saves the history JSON object to a specified S3 bucket.

    Args:
        wallet (bt.wallet): The wallet containing the hotkey used to generate the filename.
        history (Dict[str, Any]): The history data to be saved.
        bucket (str): The S3 bucket to save the history to.
        CLIENT: The client used to interact with the storage service.

    Returns:
        None
    """
    try:
        # Generate the filename based on the wallet's hotkey
        filename = f'history-{wallet.hotkey.ss58_address}.json'
        
        # Serialize the history to JSON and encode to bytes
        history_bytes = json.dumps(history).encode('utf-8')
        
        # Create a buffer from the bytes
        history_buffer = io.BytesIO(history_bytes)
        
        # Upload the history JSON to S3
        CLIENT.upload_fileobj(history_buffer, bucket, filename)
        
        # Grant read and list permissions to all users for the history file
        CLIENT.put_object_acl(
            Bucket=bucket,
            Key=filename,
            GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
            GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
        )
        
        print(f"Successfully saved history to {filename} in bucket {bucket}.")
    except Exception as e:
        print(f"Error while saving history to {filename} in bucket {bucket}: {e}")

def load_history(
        uid: int, 
        metagraph, 
        subtensor, 
        CLIENT 

    ) -> Dict[str, Any]:
    """
    Loads the history JSON object associated with the latest model from a specified S3 bucket.

    This function retrieves the latest metadata for the given uid using the subtensor and then downloads
    the corresponding history file from S3 based on the metadata.

    Args:
        uid (int): The unique identifier for the model.
        metagraph (bt.metagraph): The metagraph object used to interact with the network.
        subtensor (bt.subtensor): The subtensor object used to interact with the network.
        CLIENT: The client used to interact with the storage service.

    Returns:
        Dict[str, Any]: The loaded history data. Returns an empty dict if loading fails.
    """
    try:
        # Get the bucket name using the subtensor and metagraph information
        bucket = subtensor.get_commitment(metagraph.netuid, uid)
        hotkey = metagraph.hotkeys[ uid ]
        
        # Define the history filename based on metadata
        history_filename = f'history-{ hotkey }.json'
        
        # Create a buffer to receive the downloaded data
        history_buffer = io.BytesIO()
        
        # Download the history JSON from S3
        CLIENT.download_fileobj( bucket, history_filename, history_buffer)
        
        # Move the buffer's cursor to the beginning
        history_buffer.seek(0)
        
        # Deserialize the JSON data
        history = json.load(history_buffer)
        
        print(f"Successfully loaded history from {history_filename} in bucket { bucket}.")
        return history
    except CLIENT.exceptions.NoSuchKey:
        print(f"History file {history_filename} does not exist in bucket { bucket}. Returning empty history.")
        return {}
    except Exception as e:
        print(f"Error while loading history from {history_filename} in bucket { bucket}: {e}")
        return {}
