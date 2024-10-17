import torch
import time
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import wrap, enable_wrap
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
import gc
import os
import logging
import torch.distributed as dist
import sys

def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

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

def main():
    # Initialize distributed process group
    init_distributed()

    # Get local rank and set device
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/LLaMA-2-7B-32K",
        use_fast=False,
        clean_up_tokenization_spaces=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Initial model size
    hidden_size = 2048
    intermediate_size = hidden_size * 4
    num_hidden_layers = 16

    max_hidden_size = 0

    try:
        while True:
            clear_gpu_memory()

            if local_rank == 0:
                print(f"Attempting model with hidden_size={hidden_size}, "
                      f"intermediate_size={intermediate_size}, num_hidden_layers={num_hidden_layers}")

            model = create_and_initialize_model(hidden_size, intermediate_size, num_hidden_layers, local_rank)
            total_params = sum(p.numel() for p in model.parameters())

            if local_rank == 0:
                print(f"Total parameter count: {total_params:,}")

            try:
                # Define mixed precision policy
                mixed_precision_policy = MixedPrecision(
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

                # Move model to device before wrapping
                model.to(device)

                # Wrap the model with PyTorch's FSDP
                fsdp_params = dict(
                    mixed_precision=mixed_precision_policy,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    device_id=device,
                )

                # Use transformer_auto_wrap_policy
                # from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

                # with enable_wrap(
                #     wrapper_cls=FSDP,
                #     auto_wrap_policy=transformer_auto_wrap_policy,
                #     **fsdp_params
                # ):
                #     model = wrap(model)

                model.eval()

                # Prepare inputs
                input_text = "DeepSpeed is"
                inputs = tokenizer.encode_plus(
                    input_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                if local_rank == 0:
                    max_hidden_size = hidden_size

                # Test the model and measure tokens per second
                num_tokens = 50  # Number of tokens to generate
                start_time = time.time()

                with torch.no_grad():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=num_tokens,
                        num_return_sequences=1,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                end_time = time.time()
                generation_time = end_time - start_time
                tokens_per_second = num_tokens / generation_time

                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                if local_rank == 0:
                    print(f"Generated: {generated_text}")
                    print(f"Tokens per second: {tokens_per_second:.2f}")

                # Increase model size for next iteration
                # hidden_size = int(hidden_size * 1.1)
                # intermediate_size = int(intermediate_size * 1.1)
                # num_hidden_layers = int(num_hidden_layers * 1.1)
                # Increase model size for next iteration
                hidden_size += 256  # Increase by 256 to maintain divisibility
                num_hidden_layers += 2  # Increase number of layers
                intermediate_size = hidden_size * 4  # Update intermediate_size accordingly

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if local_rank == 0:
                        print("Out of GPU memory. Stopping.")
                    break
                else:
                    if local_rank == 0:
                        print(f"An error occurred: {e}")
                    break

            finally:
                # Explicitly free GPU memory
                del model
                torch.cuda.empty_cache()
                gc.collect()

        # Cleanup
        dist.barrier()
        dist.destroy_process_group()

    except Exception as e:
        if local_rank == 0:
            print(f"An error occurred: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    if local_rank == 0:
        print(f"Maximum model size achieved: hidden_size={max_hidden_size}")
        print(f"Using {world_size} GPUs with PyTorch FSDP for model parallelism")

if __name__ == "__main__":
    main()