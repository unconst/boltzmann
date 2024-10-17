import torch
import time
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, enable_wrap, wrap
import functools
import gc
import os
from tqdm import tqdm
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch.nn.functional as F
import typer


def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
def create_and_initialize_model(hidden_size, intermediate_size, num_hidden_layers, local_rank):
    # Calculate num_attention_heads
    num_attention_heads = max(1, hidden_size // 256)

    # Adjust hidden_size to be divisible by num_attention_heads
    if hidden_size % num_attention_heads != 0:
        hidden_size_per_head = hidden_size / num_attention_heads
        adjusted_hidden_size_per_head = round(hidden_size_per_head)
        hidden_size = num_attention_heads * adjusted_hidden_size_per_head

    # Update intermediate_size proportionally
    intermediate_size = int(hidden_size * 4)

    llama_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        hidden_act="silu"
    )

    if local_rank == 0:
        print(f"Adjusted hidden_size: {hidden_size}")
        print(f"Adjusted num_attention_heads: {num_attention_heads}")

    model = LlamaForCausalLM(llama_config)
    return model


def main(
    batch_size: int = typer.Option(1, help="Batch size for input"),
    sequence_length: int = typer.Option(2048, help="Sequence length for input"),
    steps: int = typer.Option(10, help="Number of inference steps"),
    hidden_size: int = typer.Option(4864, help="Hidden size of the model"),
    num_hidden_layers: int = typer.Option(40, help="Number of hidden layers in the model")
):
    init_distributed()

    local_rank: int = int(os.environ.get("LOCAL_RANK", "0"))
    world_size: int = dist.get_world_size()
    device: torch.device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/LLaMA-2-7B-32K",
        use_fast=False,
        clean_up_tokenization_spaces=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    clear_gpu_memory()

    intermediate_size = hidden_size * 4
    model: LlamaForCausalLM = create_and_initialize_model(hidden_size, intermediate_size, num_hidden_layers, local_rank)
    total_params: int = sum(p.numel() for p in model.parameters())

    if local_rank == 0:
        print(f"Total parameter count: {total_params:,}")

    # Define mixed precision policy
    mixed_precision_policy: MixedPrecision = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    # Define the auto wrap policy for transformer layers
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # Define FSDP parameters
    fsdp_params: dict = dict(
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=True,
        sync_module_states=False,
    )

    # Wrap the model with FSDP using the auto wrap policy
    with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        model = wrap(model)

    model.eval()

    # Prepare inputs
    input_ids: torch.Tensor = torch.randint(0, tokenizer.vocab_size, (sequence_length, batch_size), device=device)
    
    start_time: float = time.time()
    for _ in tqdm(range(steps)):
        # Use an appropriate Cache class instead of tuple for past_key_values
        outputs = model(input_ids.clone(), use_cache=True)
        past_key_values = outputs.past_key_values
        
    # Print results.
    if local_rank == 0:
        tokens = input_ids.numel() * steps
        print(f"  Time taken: {time.time() - start_time:.2f} seconds")
        print(f"  Tokens: {tokens}")
        print(f"  Tokens per second: {tokens/(time.time() - start_time):.2f}")

    # finally:
    # Ensure all processes reach this point
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    # Move model to CPU to free GPU memory
    model.cpu()

    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()

    if local_rank == 0:
        print(
            f"Completed tokens per second measurement using {world_size} GPUs with PyTorch FSDP."
        )


if __name__ == "__main__":
    typer.run(main)
