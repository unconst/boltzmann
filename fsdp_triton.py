import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from transformers import AutoTokenizer
from dataset import DatasetLoader
import asyncio
import gc
import logging
from functools import partial
from llama2_model import Transformer, ModelArgs  # Ensure this module is available

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clear_gpu_memory() -> None:
    """
    Clears GPU memory cache and performs garbage collection.

    This function frees up GPU memory by clearing the CUDA cache and
    invoking garbage collection to remove unreferenced objects.
    """
    logger.info("Clearing GPU memory")
    torch.cuda.empty_cache()
    gc.collect()


def setup_distributed() -> tuple[int, int, int]:
    """
    Sets up the distributed environment.

    Initializes the process group for distributed training using NCCL backend,
    sets the local device, and returns the local rank, global rank, and world size.

    Returns:
        tuple[int, int, int]: A tuple containing local_rank, rank, and world_size.
    """
    dist.init_process_group(backend="nccl")
    rank: int = dist.get_rank()
    world_size: int = dist.get_world_size()
    local_rank: int = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if rank == 0:
        logger.info(f"Distributed setup complete. World Size: {world_size}")
    return local_rank, rank, world_size


def cleanup() -> None:
    """
    Performs cleanup operations.

    Destroys the process group to ensure that all pending NCCL operations
    are completed before the script exits.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Process group destroyed")


async def main() -> None:
    """
    The main function for distributed training.

    This asynchronous function sets up the distributed environment, initializes
    the model, tokenizer, optimizer, and dataset, and runs the training loop.
    """
    if "LOCAL_RANK" not in os.environ:
        print("An error occurred: LOCAL_RANK environment variable not set")
        sys.exit(1)

    # Wrap setup in try-except to handle initialization errors
    try:
        local_rank, rank, world_size = setup_distributed()
    except Exception as e:
        print(f"An error occurred during setup_distributed: {e}")
        sys.exit(1)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if rank == 0:
        logger.info(f"Using device: {device}")

    # Set backend options for performance optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if rank == 0:
        logger.info("Backend options set for performance")

    # Load the tokenizer
    if rank == 0:
        logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/LLaMA-2-7B-32K",
        use_fast=False,
        clean_up_tokenization_spaces=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    if rank == 0:
        logger.info("Tokenizer loaded successfully")

    # Model configuration
    if rank == 0:
        logger.info("Setting up model configuration")
    model_args = ModelArgs(
        dim=4096,
        n_layers=64,
        n_heads=32,
        n_kv_heads=32,
        vocab_size=32000,
        max_seq_len=1024,
    )
    if rank == 0:
        logger.info(f"Model configuration: {model_args}")

    # Create the model
    if rank == 0:
        logger.info("Creating model")
    model = Transformer(model_args)
    model.to(device)
    model.init_weights()
    if rank == 0:
        logger.info("Model created and initialized")

        # Calculate and log the total number of parameters in the model
        total_params: int = sum(param.numel() for param in model.parameters())
        logger.info(f"Model has {total_params} parameters")

    # Configure FSDP settings
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,  # Data type for model parameters
        reduce_dtype=torch.float16,  # Data type for gradient reduction
        buffer_dtype=torch.float16,  # Data type for other buffers
    )

    # Define the Transformer layer class for auto-wrapping
    from llama2_model_2 import TransformerBlock  # Ensure correct import

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # Apply activation checkpointing to reduce memory consumption
    non_reentrant_wrapper = partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: isinstance(submodule, TransformerBlock),
    )
    if rank == 0:
        logger.info("Activation checkpointing applied")

    # Wrap the model with FSDP using the auto-wrap policy and mixed precision
    if rank == 0:
        logger.info("Wrapping model with FSDP")
    fsdp_params = dict(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=False),
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=device,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
    )
    model = FSDP(model, **fsdp_params)
    if rank == 0:
        logger.info("Model wrapped with FSDP")

    # Compile the FSDP-wrapped model using torch.compile
    if rank == 0:
        logger.info("Compiling the model with torch.compile")

    model = torch.compile(model)

    # Prepare the optimizer
    if rank == 0:
        logger.info("Preparing optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if rank == 0:
        logger.info("Optimizer prepared")

    # Prepare the dataset
    batch_size: int = 32  # Adjusted for memory constraints
    sequence_length: int = 2048
    if rank == 0:
        logger.info(
            f"Preparing data with batch size: {batch_size}, sequence length: {sequence_length}"
        )

    # Load the dataset asynchronously
    pages = await DatasetLoader.next_pages(offset=1, n_pages=1, seed=1)
    dataset = await DatasetLoader.create(
        batch_size=batch_size,
        sequence_length=sequence_length,
        pages_info=pages,
        tokenizer=tokenizer,
    )
    if rank == 0:
        logger.info("Dataset prepared")

    # Initialize gradient scaler for mixed-precision training
    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    num_steps: int = -1  # Set to a positive integer to limit steps
    total_tokens: int = 0
    start_time: float = time.time()
    if rank == 0:
        logger.info("Starting training loop")

    for step, batch in enumerate(dataset):
        if step >= num_steps and num_steps != -1:
            if rank == 0:
                logger.info(f"Reached step limit of {num_steps}. Stopping training.")
            break

        # Prepare input data
        input_ids = torch.tensor(batch, dtype=torch.long).to(device)
        labels = input_ids.clone()
        labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
        if rank == 0:
            logger.info(f"Step {step}: Input prepared")

        # Forward pass with mixed precision
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids)
            loss_fn = nn.CrossEntropyLoss()
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            labels_flat = labels.view(-1)
            loss = loss_fn(outputs_flat, labels_flat)
        if rank == 0:
            logger.info(f"Step {step}: Forward pass completed. Loss: {loss.item():.4f}")

        # Backward pass and optimization with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if rank == 0:
            logger.info(f"Step {step}: Backward pass completed")

        # Update token count
        total_tokens += input_ids.numel()
        if rank == 0:
            logger.info(
                f"Step {step}: Processed {input_ids.numel()} tokens. "
                f"Total tokens: {total_tokens}"
            )

    elapsed_time: float = time.time() - start_time
    if rank == 0:
        logger.info("Training loop completed")
        logger.info(f"Time taken: {elapsed_time:.2f} seconds")
        logger.info(f"Tokens processed: {total_tokens}")
        logger.info(f"Tokens per second: {total_tokens / elapsed_time:.2f}")

    # Cleanup operations
    cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info("Script completed successfully")
    except Exception as e:
        print(f"An error occurred: {e}")
        cleanup()
        sys.exit(1)
