import torch
import time
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, enable_wrap, wrap
import functools
import gc
import os
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch.nn.functional as F


def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def create_12b_model(local_rank: int) -> LlamaForCausalLM:
    # Configuration for a ~12B parameter model
    hidden_size: int = 5632
    num_attention_heads: int = 22  # Adjusted to match hidden_size
    num_hidden_layers: int = 44
    intermediate_size: int = hidden_size * 4

    llama_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads // 2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        hidden_act="silu",
    )

    if local_rank == 0:
        print(f"Creating model with hidden_size: {hidden_size}")
        print(f"Number of attention heads: {num_attention_heads}")
        print(f"Number of hidden layers: {num_hidden_layers}")

    # Initialize the model on CPU
    model = LlamaForCausalLM(llama_config)
    return model


def main():
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

    if local_rank == 0:
        print("Initializing the 12B parameter model to measure tokens per second.")

    model: LlamaForCausalLM = create_12b_model(local_rank)
    total_params: int = sum(p.numel() for p in model.parameters())

    if local_rank == 0:
        print(f"Total parameter count: {total_params:,}")

    try:
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
        batch_size: int = 50
        input_text: list = ["DeepSpeed is a machine learning framework"] * batch_size
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids: torch.Tensor = inputs["input_ids"].to(device)
        attention_mask: torch.Tensor = inputs["attention_mask"].to(device)

        # Variables to measure performance
        num_tokens_to_generate: int = 50
        num_iterations: int = 5
        total_generated_tokens: int = 0
        total_time: float = 0.0

        # Set temperature for sampling
        temperature = 1.0  # Adjust this value for more or less randomness

        # Manually generate tokens
        for iteration in range(1, num_iterations + 1):
            torch.cuda.synchronize()
            start_time: float = time.time()

            with torch.no_grad():
                generated_ids = input_ids.clone()
                generated_attention_mask = attention_mask.clone()

                for _ in range(num_tokens_to_generate):
                    outputs = model(
                        generated_ids,
                        attention_mask=generated_attention_mask,
                    )
                    next_token_logits = outputs.logits[:, -1, :]

                    # Apply temperature
                    next_token_logits = next_token_logits / temperature

                    # Convert logits to probabilities
                    next_token_probs = torch.softmax(next_token_logits, dim=-1)

                    # Sample the next token
                    next_token = torch.multinomial(next_token_probs, num_samples=1)

                    # Append the new token to the generated sequence
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                    # Update the attention mask
                    next_token_attention = torch.ones(
                        (generated_attention_mask.size(0), 1),
                        dtype=generated_attention_mask.dtype,
                    ).to(device)
                    generated_attention_mask = torch.cat(
                        [generated_attention_mask, next_token_attention], dim=1
                    )

            torch.cuda.synchronize()
            end_time: float = time.time()

            iteration_time: float = end_time - start_time
            generated_tokens: int = num_tokens_to_generate * batch_size
            tokens_per_second: float = generated_tokens / iteration_time

            total_generated_tokens += generated_tokens
            total_time += iteration_time

            if local_rank == 0:
                print(f"Iteration {iteration}:")
                print(f"  Time taken: {iteration_time:.2f} seconds")
                print(f"  Tokens generated: {generated_tokens}")
                print(f"  Tokens per second: {tokens_per_second:.2f}")

        avg_tokens_per_second: float = total_generated_tokens / total_time

        if local_rank == 0:
            # Decode the last output as an example
            generated_text: list = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            print("\nGenerated Text Example:")
            for idx, text in enumerate(generated_text):
                print(f"Sample {idx + 1}: {text}\n")
            print(
                f"Average Tokens per Second over {num_iterations} iterations: {avg_tokens_per_second:.2f}"
            )

    except Exception as e:
        if local_rank == 0:
            print(f"An error occurred: {e}")

    finally:
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
    main()
