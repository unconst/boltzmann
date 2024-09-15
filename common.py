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
from botocore.exceptions import ClientError  # Import for handling S3 client errors

def human_readable_size(size, decimal_places=2):
    """
    Converts a size in bytes to a human-readable string format.

    Args:
        size (float): The size in bytes.
        decimal_places (int): Number of decimal places to include.

    Returns:
        str: A string representing the size in appropriate units (B, KB, MB, GB, TB).
    """
    # Iterate through each unit of measurement.
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        # If the size is less than 1024, we have found the appropriate unit.
        if size < 1024.0:
            # Return the size formatted with the unit.
            return f"{size:.{decimal_places}f} {unit}"
        # Otherwise, divide the size by 1024 and move to the next unit.
        size /= 1024.0

def get_latest_metadata_block(hotkey: str, bucket: str, CLIENT) -> int:
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
    # List all objects in the specified bucket.
    response = CLIENT.list_objects_v2(Bucket=bucket)
    # TODO: Handle pagination if there are more objects than the max keys returned.

    # Extract the filenames from the response.
    file_names = [content['Key'] for content in response.get('Contents', [])]

    max_block = -1  # Initialize the maximum block number found.
    latest_file = None  # Placeholder for the latest file.

    # Iterate over all filenames.
    for file_name in file_names:
        # Check if the filename matches the model pattern for the given hotkey.
        if file_name.startswith(f'model-{hotkey}-') and file_name.endswith('.pt'):
            try:
                # Extract the block number from the filename.
                block = int(file_name.split('-')[-1].split('.')[0])
                # Update max_block if a newer block is found.
                if block > max_block:
                    max_block = block
                    latest_file = file_name
            except ValueError:
                # Skip files that don't have a valid block number.
                continue
    # Return the highest block number found.
    return max_block

def hash_model(module: torch.nn.Module) -> str:
    """
    Generates a SHA-256 hash of the model's state dictionary.

    This function iterates through the model's state dictionary, concatenates the byte representation
    of each parameter, and then generates a SHA-256 hash of this concatenated byte string.

    Args:
        module (torch.nn.Module): The model to hash.

    Returns:
        str: The SHA-256 hash of the model's state dictionary.
    """
    if module is None:
        # Return a default hash if the module is None.
        return "0x0000000000000000000000000000000000000000000000000000000000000000"
    
    # Get the state dictionary containing all model parameters.
    module_state_dict = module.state_dict()
    
    # Concatenate all parameter tensors into a single byte string.
    concatenated_model_states_bytes = b''.join(
        [value.cpu().numpy().tobytes() for value in module_state_dict.values()]
    )
    
    # Compute the SHA-256 hash of the concatenated bytes.
    module_hash = hashlib.sha256(concatenated_model_states_bytes).hexdigest()
    return module_hash

def get_latest_metadata_for_hotkey_and_bucket(
        hotkey: str,
        bucket: str,
        CLIENT
    ) -> Optional[SimpleNamespace]:
    """
    Retrieves the latest metadata for a given hotkey from a specified bucket.

    This function finds the latest model file for the given hotkey in the specified bucket,
    retrieves its metadata, and combines it with additional metadata from a corresponding JSON file.

    Args:
        hotkey (str): The hotkey associated with the model files.
        bucket (str): The name of the bucket where the model files are stored.
        CLIENT: The client used to interact with the storage service.

    Returns:
        Optional[SimpleNamespace]: An object containing combined metadata from the model file and its metadata JSON.
                                   Returns None if no model is found for the given hotkey.
    """
    # Get the latest block number for the hotkey.
    latest_block = get_latest_metadata_block(hotkey, bucket, CLIENT)
    if latest_block == -1:
        # If no model files are found, return None.
        return None
    
    # Define filenames for the model and its metadata based on the latest block.
    filename = f"model-{hotkey}-{latest_block}.pt"
    metadata_filename = f"model-{hotkey}-{latest_block}_metadata.json"
    
    # Get the metadata of the model file from the storage service.
    response = CLIENT.head_object(Bucket=bucket, Key=filename)
    
    # Extract and calculate metadata information.
    metadata = {}
    metadata['last_modified'] = int(response['LastModified'].timestamp())
    # Estimate blocks since modification assuming 12 seconds per block.
    metadata['blocks_since_modified'] = int((time.time() - metadata['last_modified']) / 12)
    metadata['size'] = response['ContentLength']
    metadata['bucket'] = bucket
    metadata['filename'] = filename
    metadata['metadata_filename'] = metadata_filename

    # Get the metadata file from the storage service.
    metadata_response = CLIENT.get_object(Bucket=bucket, Key=metadata_filename)
    
    # Read and update the metadata with the content of the metadata file.
    metadata_json = json.loads(metadata_response['Body'].read().decode('utf-8'))
    metadata.update(metadata_json)
    
    # Return the metadata as a SimpleNamespace for easy attribute access.
    return SimpleNamespace(**metadata)

def get_latest_metadata( 
        uid: int, 
        metagraph, 
        subtensor, 
        CLIENT 
    ) -> Optional[SimpleNamespace]:
    """
    Retrieves the latest metadata for a specified UID from the storage service.

    This function uses the metagraph and subtensor to determine the bucket and hotkey
    associated with the given UID, then retrieves the latest model metadata.

    Args:
        uid (int): The unique identifier for the model.
        metagraph: The Bittensor metagraph containing network information.
        subtensor: The Bittensor subtensor object used to interact with the network.
        CLIENT: The client used to interact with the storage service.

    Returns:
        Optional[SimpleNamespace]: An object containing the metadata if successful, otherwise None.
    """
    try:
        # Get the bucket name (commitment) for the given netuid and UID.
        bucket = subtensor.get_commitment(metagraph.netuid, uid)
        # Get the hotkey associated with the UID from the metagraph.
        hotkey = metagraph.hotkeys[uid]
        # Retrieve the latest metadata for the hotkey and bucket.
        metadata = get_latest_metadata_for_hotkey_and_bucket(hotkey=hotkey, bucket=bucket, CLIENT=CLIENT)
        if metadata is not None:
            # Add the UID to the metadata.
            metadata.uid = int(uid)
            return metadata
        else: 
            # Return None if no metadata is found.
            return None
    except Exception as e:
        # Print the exception message.
        print(e)
        # Return None if any exception occurs.
        return None

def upload_model( 
        wallet: 'bt.wallet',
        model: torch.nn.Module, 
        block: int,
        extras: Dict[str, object],
        bucket: str,
        CLIENT,
    ) -> SimpleNamespace:
    """
    Uploads a model to a specified bucket along with its metadata.

    Args:
        wallet (bt.wallet): The wallet containing the hotkey used to generate the filename.
        model (torch.nn.Module): The model to be uploaded.
        block (int): The block number associated with the model.
        extras (Dict[str, object]): Additional metadata to be uploaded with the model.
        bucket (str): The bucket to upload the model to.
        CLIENT: The client used to interact with the storage service.

    Returns:
        SimpleNamespace: The metadata of the uploaded model.
    """
    start_time = time.time()  # Record the start time for the upload process.
    model_state_dict = model.state_dict()  # Get the state dictionary of the model.

    # Extract the configuration from the model and update extras with model type and configuration.
    if isinstance(model, LlamaForCausalLM):
        config = model.config  # Get the configuration of the Llama model.
        extras.update({
            'model_type': 'llama',  # Add model type to extras.
            'model_config': config.to_dict()  # Add model configuration to extras.
        })
    elif isinstance(model, GPT2LMHeadModel):
        config = model.config  # Get the configuration of the GPT-2 model.
        extras.update({
            'model_type': 'gpt2',  # Add model type to extras.
            'model_config': config.to_dict()  # Add model configuration to extras.
        })
    else:
        # If the model type is not recognized, raise an exception.
        raise ValueError("Unsupported model type for uploading.")

    # Add the hash of the model to the metadata.
    extras['model_hash'] = hash_model(model)

    # Generate filenames for the model and its metadata based on the hotkey and block number.
    filename = f'model-{wallet.hotkey.ss58_address}-{block}.pt'  # Filename for the model.
    metadata_filename = f"model-{wallet.hotkey.ss58_address}-{block}_metadata.json"  # Filename for the metadata.

    # Upload the metadata to the storage service.
    # Convert the extras dictionary to JSON and encode it to bytes.
    metadata_buffer = io.BytesIO(json.dumps(extras).encode('utf-8'))  # Create a buffer for the metadata.
    # Upload the metadata buffer to the storage service.
    CLIENT.upload_fileobj(metadata_buffer, bucket, metadata_filename)

    # Grant read and list permissions to all users for the metadata file.
    CLIENT.put_object_acl(
        Bucket=bucket,
        Key=metadata_filename,
        GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
        GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
    )
    
    # Upload the model to the storage service.
    with io.BytesIO() as module_buffer:
        # Save the model state dictionary to the buffer.
        torch.save(model_state_dict, module_buffer)
        module_buffer.seek(0)  # Reset the buffer's position to the beginning.
        # Upload the model buffer to the storage service.
        CLIENT.upload_fileobj(module_buffer, bucket, filename)

    # Grant read and list permissions to all users for the model file.
    CLIENT.put_object_acl(
        Bucket=bucket,
        Key=filename,
        GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
        GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
    )

    # Retrieve the metadata of the uploaded model.
    returned_metadata = get_latest_metadata_for_hotkey_and_bucket(
        hotkey=wallet.hotkey.ss58_address, bucket=bucket, CLIENT=CLIENT
    )
    # Log the completion of the upload process with the time taken.
    print(f"Uploaded model to {filename}@{bucket} of size: {human_readable_size(returned_metadata.size)} in: {time.time() - start_time} seconds.")
    return returned_metadata

def download_model( 
        metadata: SimpleNamespace, 
        device: str, 
        CLIENT,
    ) -> Optional[torch.nn.Module]:
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
        # Log the start of the download process.
        print(f'Downloading model from {metadata.filename}@{metadata.bucket}')
        start_time = time.time()  # Record the start time for the download.

        # Check the model type and initialize the appropriate model configuration and model.
        if metadata.model_type == "llama":
            # Create Llama model configuration from metadata.
            model_config = LlamaConfig(**metadata.model_config)
            # Initialize Llama model with the configuration.
            model = LlamaForCausalLM(model_config)
        elif metadata.model_type == "gpt2":
            # Create GPT-2 model configuration from metadata.
            model_config = GPT2Config(**metadata.model_config)
            # Initialize GPT-2 model with the configuration.
            model = GPT2LMHeadModel(model_config)
        else:
            # If model type is unknown, raise an exception.
            raise ValueError(f"Unsupported model type: {metadata.model_type}")
            
        # Generate a unique temporary file path using uuid.
        unique_temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pt")

        # Download the model file from the storage service to the temporary file.
        CLIENT.download_file(metadata.bucket, metadata.filename, unique_temp_file)

        # Load the model state dict from the temporary file.
        new_model_state_dict = torch.load(unique_temp_file, map_location=torch.device(device))
        
        # Load the state dict into the model.
        model.load_state_dict(new_model_state_dict)
        # Move the model to the specified device.
        model.to(device)

        # Remove the temporary file after use.
        os.remove(unique_temp_file)

        # Log the completion of the download process with the time taken.
        print(f"Downloaded model from {metadata.filename}@{metadata.bucket} of size: {human_readable_size(metadata.size)} in {time.time() - start_time} seconds.")
        # Return the downloaded model.
        return model
    except Exception as e:
        # Log any exceptions that occur during the download.
        print(f'Error while downloading model from {metadata.filename}@{metadata.bucket} with error {e}.')
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
        # Generate the filename for the history file based on the wallet's hotkey.
        filename = f'history-{wallet.hotkey.ss58_address}.json'
        
        # Serialize the history dictionary to a JSON-formatted string and encode it to bytes.
        history_bytes = json.dumps(history).encode('utf-8')
        
        # Create a buffer from the bytes to upload.
        history_buffer = io.BytesIO(history_bytes)
        
        # Upload the history JSON to the specified S3 bucket.
        CLIENT.upload_fileobj(history_buffer, bucket, filename)
        
        # Grant read and list permissions to all users for the history file.
        CLIENT.put_object_acl(
            Bucket=bucket,
            Key=filename,
            GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
            GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
        )
        
        # Log a success message.
        print(f"Successfully saved history to {filename} in bucket {bucket}.")
    except Exception as e:
        # Log any exceptions that occur during the save process.
        print(f"Error while saving history to {filename} in bucket {bucket}: {e}")

def load_history(
        uid: int, 
        metagraph, 
        subtensor, 
        CLIENT 
    ) -> Dict[str, Any]:
    """
    Loads the history JSON object associated with the latest model from a specified S3 bucket.

    This function retrieves the latest metadata for the given UID using the subtensor and then downloads
    the corresponding history file from S3 based on the metadata.

    Args:
        uid (int): The unique identifier for the model.
        metagraph: The metagraph object used to interact with the network.
        subtensor: The subtensor object used to interact with the network.
        CLIENT: The client used to interact with the storage service.

    Returns:
        Dict[str, Any]: The loaded history data. Returns an empty dict if loading fails.
    """
    try:
        # Get the bucket name (commitment) for the given netuid and UID.
        bucket = subtensor.get_commitment(metagraph.netuid, uid)
        # Get the hotkey associated with the UID from the metagraph.
        hotkey = metagraph.hotkeys[uid]
        
        # Define the history filename based on the hotkey.
        history_filename = f'history-{hotkey}.json'
        
        # Create a buffer to receive the downloaded data.
        history_buffer = io.BytesIO()
        
        # Download the history JSON file from S3 into the buffer.
        CLIENT.download_fileobj(Bucket=bucket, Key=history_filename, Fileobj=history_buffer)
        
        # Move the buffer's cursor to the beginning.
        history_buffer.seek(0)
        
        # Deserialize the JSON data from the buffer.
        history = json.load(history_buffer)
        
        # Log a success message.
        print(f"Successfully loaded history from {history_filename} in bucket {bucket}.")
        return history
    except ClientError as e:
        # Check if the error is a 'NoSuchKey' error.
        if e.response['Error']['Code'] == 'NoSuchKey':
            # If the history file does not exist, return an empty history.
            print(f"History file {history_filename} does not exist in bucket {bucket}. Returning empty history.")
            return {}
        else:
            # For other client errors, log the error and return empty history.
            print(f"ClientError while loading history from {history_filename} in bucket {bucket}: {e}")
            return {}
    except Exception as e:
        # Log any other exceptions that occur during the load process.
        print(f"Error while loading history from {history_filename} in bucket {bucket}: {e}")
        return {}
