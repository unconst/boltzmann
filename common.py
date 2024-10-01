import os
import io
import sys 
import uuid
import torch
import uvloop
import hashlib
import asyncio
import tempfile
import aiofiles
import numpy as np
import aiobotocore
import bittensor as bt
import botocore.config
from loguru import logger
from typing import List, Dict
from dotenv import dotenv_values
from types import SimpleNamespace
from aiobotocore.session import get_session

# Configure loguru logger
logger.remove()
logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")

# Load environment variables
env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')

# Configure the S3 client
client_config = botocore.config.Config(
    max_pool_connections=256,
)

# Set uvloop as the event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Define a semaphore to limit concurrent downloads (adjust as needed)
semaphore = asyncio.Semaphore(1000)

async def upload_mask(bucket: str, model: torch.nn.Module, mask: int, wallet: 'bt.wallet', compression: int):
    """
    Uploads a compressed mask of a PyTorch model to an S3 bucket.

    Args:
        bucket (str): Name of the S3 bucket.
        model (torch.nn.Module): The PyTorch model to be masked and uploaded.
        mask (int): The mask identifier.
        wallet (bt.wallet): The wallet object containing the hotkey.
        compression (int): The compression factor.
    """
    filename = f'mask-{mask}-{wallet.hotkey.ss58_address}.pt'
    logger.debug(f"Uploading mask to S3: {filename}")

    model_state_dict = model.state_dict()
    indices = await get_indices_for_mask(model, mask, compression)

    # Apply the mask to the model parameters
    for name, param in model.named_parameters():
        model_state_dict[name] = param.data.view(-1)[indices[name].to(model.device)].cpu()

    # Create a temporary file and write the masked model state dictionary to it
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        torch.save(model_state_dict, temp_file)
        temp_file_name = temp_file.name  # Store the temporary file name

    # Upload the file to S3
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        try:
            with open(temp_file_name, 'rb') as f:
                await s3_client.put_object(Bucket=bucket, Key=filename, Body=f)
            # Set the object ACL to public-read
            await s3_client.put_object_acl(
                Bucket=bucket,
                Key=filename,
                ACL='public-read'
            )
            logger.debug(f"Successfully uploaded mask to S3: {filename}")
        except Exception:
            logger.exception(f"Failed to upload mask {filename} to S3")
        finally:
            # Clean up the temporary file
            os.remove(temp_file_name)
            logger.debug(f"Temporary file {temp_file_name} removed")

async def upload_master(bucket: str, model: torch.nn.Module, wallet: 'bt.wallet'):
    """
    Uploads the master PyTorch model to an S3 bucket.

    Args:
        bucket (str): Name of the S3 bucket.
        model (torch.nn.Module): The PyTorch model to be uploaded.
        wallet (bt.wallet): The wallet object containing the hotkey.
    """
    upload_filename = f'master-{wallet.hotkey.ss58_address}.pt'
    logger.debug(f"Uploading master model to S3: {upload_filename}")

    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        try:
            # Create a temporary file and write the model state dictionary to it
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                torch.save(model.state_dict(), temp_file)
                temp_file_name = temp_file.name

            # Upload the file to S3
            with open(temp_file_name, 'rb') as f:
                await s3_client.put_object(Bucket=bucket, Key=upload_filename, Body=f)
            # Set the object ACL to public-read
            await s3_client.put_object_acl(
                Bucket=bucket,
                Key=upload_filename,
                ACL='public-read'
            )
            logger.debug(f"Successfully uploaded master model to S3: {upload_filename}")
        except Exception:
            logger.exception(f"Failed to upload master model {upload_filename} to S3")
        finally:
            # Clean up the temporary file
            os.remove(temp_file_name)
            logger.debug(f"Temporary file {temp_file_name} removed")

async def get_indices_for_mask(model: torch.nn.Module, mask: int, compression: int) -> Dict[str, torch.LongTensor]:
    """
    Computes the indices for the given mask and compression factor.

    Args:
        model (torch.nn.Module): The PyTorch model.
        mask (int): The mask identifier.
        compression (int): The compression factor.

    Returns:
        Dict[str, torch.LongTensor]: A dictionary mapping parameter names to index tensors.
    """
    logger.debug(f"Computing indices for mask {mask} with compression {compression}")
    result = {}
    # Seed the random number generator with the mask
    seed = int(hashlib.md5(str(mask).encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    for name, param in sorted(model.named_parameters()):
        # Randomly select indices based on the compression factor
        num_indices = max(1, int(param.numel() // compression))
        indices = rng.choice(param.numel(), size=num_indices, replace=False)
        result[name] = torch.from_numpy(indices).long().cpu()
    return result

async def download_file(s3_client, bucket: str, filename: str) -> str:
    """
    Downloads a file from S3, using parallel downloads for large files.

    Args:
        s3_client: The S3 client.
        bucket (str): Name of the S3 bucket.
        filename (str): The S3 object key (filename).

    Returns:
        str: The path to the downloaded file in the temporary directory.
    """
    async with semaphore:
        try:
            temp_file = os.path.join(tempfile.gettempdir(), filename)

            # Check if the file already exists
            if os.path.exists(temp_file):
                logger.debug(f"File {temp_file} already exists, skipping download.")
                return temp_file

            # Get the object size
            head_response = await s3_client.head_object(Bucket=bucket, Key=filename)
            object_size = head_response['ContentLength']

            # Define the chunk size and calculate the number of parts
            CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB
            part_size = CHUNK_SIZE
            parts = []
            for i in range(0, object_size, part_size):
                start = i
                end = min(i + part_size - 1, object_size - 1)
                parts.append((start, end))

            # Function to download a part
            async def download_part(part_number, start, end):
                try:
                    range_header = f"bytes={start}-{end}"
                    response = await s3_client.get_object(
                        Bucket=bucket,
                        Key=filename,
                        Range=range_header
                    )

                    # Check the HTTP status code
                    status_code = response['ResponseMetadata']['HTTPStatusCode']
                    if status_code != 206:
                        logger.error(f"Unexpected status code {status_code} for part {part_number}")
                        raise Exception(f"Unexpected status code {status_code}")

                    part_file = f"{temp_file}.part{part_number}"
                    async with aiofiles.open(part_file, 'wb') as f:
                        while True:
                            chunk = await response['Body'].read(CHUNK_SIZE)
                            if not chunk:
                                break
                            await f.write(chunk)
                    logger.trace(f"Downloaded part {part_number} to {part_file}")
                    return part_file
                except Exception as e:
                    logger.error(f"Error downloading part {part_number}: {e}")
                    raise

            # Download parts concurrently
            download_tasks = [
                download_part(idx, start, end)
                for idx, (start, end) in enumerate(parts)
            ]

            part_files = await asyncio.gather(*download_tasks)

            # Combine parts into the final file
            async with aiofiles.open(temp_file, 'wb') as outfile:
                for part_file in sorted(part_files, key=lambda x: int(x.split('part')[-1])):
                    async with aiofiles.open(part_file, 'rb') as infile:
                        while True:
                            chunk = await infile.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            await outfile.write(chunk)
                    # Remove the part file
                    try:
                        os.remove(part_file)
                        logger.debug(f"Removed part file {part_file}")
                    except OSError as e:
                        logger.error(f"Error removing part file {part_file}: {e}")

            logger.debug(f"Successfully downloaded file {filename} to {temp_file}")
            return temp_file
        except Exception as e:
            logger.exception(f"Failed to download file {filename} from bucket {bucket}: {e}")
            return None

async def handle_file(s3_client, bucket: str, filename: str, hotkey: str, mask: int):
    """
    Handles downloading a single file from S3.

    Args:
        s3_client: The S3 client.
        bucket (str): Name of the S3 bucket.
        filename (str): The S3 object key (filename).
        hotkey (str): The hotkey identifier.
        mask (int): The mask identifier.

    Returns:
        SimpleNamespace: An object containing file metadata and the path to the downloaded file.
    """
    logger.debug(f"Handling file {filename} for mask {mask} and hotkey {hotkey}")
    temp_file = await download_file(s3_client, bucket, filename)
    if temp_file:
        return SimpleNamespace(bucket=bucket, hotkey=hotkey, filename=filename, mask=mask, temp_file=temp_file)
    return None

async def process_bucket(s3_client, bucket: str, masks: List[int]):
    """
    Processes an S3 bucket to download files matching the given masks.

    Args:
        s3_client: The S3 client.
        bucket (str): Name of the S3 bucket.
        masks (List[int]): A list of mask identifiers.

    Returns:
        List[SimpleNamespace]: A list of file metadata and paths for downloaded files.
    """
    logger.debug(f"Processing bucket {bucket} for masks {masks}")
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')

    for mask in masks:
        prefix = f'mask-{mask}'
        logger.debug(f"Listing objects with prefix {prefix}")
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            logger.trace(f"Processing page for prefix {prefix}")
            if 'Contents' not in page:
                logger.trace(f"No contents found for prefix {prefix}")
                continue
            download_tasks = []
            for obj in page.get('Contents', []):
                filename = obj['Key']
                logger.trace(f"Processing object with key {filename}")
                try:
                    parts = filename.split('-')
                    file_mask = int(parts[1])
                    hotkey = parts[2].split('.')[0]
                    logger.trace(f"Parsed filename {filename} into mask {file_mask} and hotkey {hotkey}")
                    if file_mask == mask:
                        download_tasks.append(handle_file(s3_client, bucket, filename, hotkey, file_mask))
                except Exception:
                    logger.exception(f"Error processing filename {filename}")
                    continue
            # Download the files concurrently
            results = await asyncio.gather(*download_tasks)
            files.extend([res for res in results if res])
            logger.trace(f"Completed processing page for prefix {prefix}")
    logger.trace(f"Completed processing bucket {bucket} for masks {masks}")
    return files

async def download_files_for_buckets_and_masks(buckets: List[str], masks: List[int]) -> Dict[int, List[SimpleNamespace]]:
    """
    Downloads files from multiple S3 buckets for the given masks.

    Args:
        buckets (List[str]): A list of S3 bucket names.
        masks (List[int]): A list of mask identifiers.

    Returns:
        Dict[int, List[SimpleNamespace]]: A dictionary mapping masks to lists of file metadata and paths.
    """
    logger.debug(f"Downloading files for buckets {set(buckets)} and masks {masks}")
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        tasks = []
        for bucket in set(buckets):
            if not bucket:
                continue
            tasks.append(process_bucket(s3_client, bucket, masks))
        results = await asyncio.gather(*tasks)
        # Flatten the list of lists
        files = [item for sublist in results for item in sublist]

        # Create a dictionary with masks as keys and list of files as values
        mask_dict = {}
        for file in files:
            mask = file.mask
            if mask not in mask_dict:
                mask_dict[mask] = []
            mask_dict[mask].append(file)

        logger.debug(f"Downloaded files grouped by mask: {list(mask_dict.keys())}")
        return mask_dict

async def get_files_for_mask_from_temp(mask: int) -> List[str]:
    """
    Retrieves the paths to downloaded mask files from the temporary directory.

    Args:
        mask (int): The mask identifier.

    Returns:
        List[str]: A list of file paths corresponding to the mask.
    """
    logger.debug(f"Retrieving files for mask {mask} from temporary directory")
    temp_dir = tempfile.gettempdir()
    mask_files = []
    for filename in os.listdir(temp_dir):
        if filename.startswith(f"mask-{mask}-") and filename.endswith(".pt"):
            mask_files.append(os.path.join(temp_dir, filename))
            logger.debug(f"Found file {filename} for mask {mask}")
    return mask_files

async def delete_files_before_mask(mask_max: int):
    """
    Deletes all files on the local machine which have a mask id before a specific value mask_max.

    Args:
        mask_max (int): The maximum mask id. Files with mask ids less than this value will be deleted.
    """
    logger.debug(f"Deleting files with mask id before {mask_max}")
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith("mask-") and filename.endswith(".pt"):
            try:
                parts = filename.split('-')
                mask_id = int(parts[1])
                if mask_id < mask_max:
                    file_path = os.path.join(temp_dir, filename)
                    os.remove(file_path)
                    logger.debug(f"Deleted file {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {filename}: {e}")

async def delete_files_from_bucket_before_mask(bucket: str, mask_max: int):
    """
    Deletes all files in the specified S3 bucket which have a mask id before a specific value mask_max.

    Args:
        bucket (str): The name of the S3 bucket.
        mask_max (int): The maximum mask id. Files with mask ids less than this value will be deleted.
    """
    logger.debug(f"Deleting files in bucket {bucket} with mask id before {mask_max}")
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        try:
            response = await s3_client.list_objects_v2(Bucket=bucket)
            if 'Contents' in response:
                for obj in response['Contents']:
                    filename = obj['Key']
                    if filename.startswith("mask-") and filename.endswith(".pt"):
                        try:
                            parts = filename.split('-')
                            mask_id = int(parts[1])
                            if mask_id < mask_max:
                                await s3_client.delete_object(Bucket=bucket, Key=filename)
                                logger.debug(f"Deleted file {filename} from bucket {bucket}")
                        except Exception as e:
                            logger.error(f"Error deleting file {filename} from bucket {bucket}: {e}")
        except Exception as e:
            logger.error(f"Error listing objects in bucket {bucket}: {e}")
