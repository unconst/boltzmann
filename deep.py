import torch
import time
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
import deepspeed
import gc
import os
import logging

# Set the DeepSpeed log level to ERROR
os.environ['DEEPSPEED_LOG_LEVEL'] = 'ERROR'

# Set all loggers to ERROR level
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Specifically set DeepSpeed logger to ERROR level
logging.getLogger("deepspeed").setLevel(logging.ERROR)

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def create_and_initialize_model(hidden_size, intermediate_size, num_hidden_layers, ds_config):
    from deepspeed import zero

    with zero.Init(config_dict_or_path=ds_config):
        num_attention_heads = max(1, hidden_size // 64)  # Ensure divisibility
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
            activation_function="swiGLU"
        )

        model = LlamaForCausalLM(llama_config)
    return model

def main():
    # Remove manual initialization of the distributed process group
    # DeepSpeed launcher handles this automatically
    # deepspeed.init_distributed()  # Not needed when using deepspeed launcher

    # Get local rank and set device
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Configure DeepSpeed for ZeRO Stage 3
    ds_config = {
        "train_batch_size": num_gpus,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 3},
        "logging": { "level": "ERROR" },
    }

    tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/LLaMA-2-7B-32K", verbose=False, clean_up_tokenization_spaces=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Initial model size
    hidden_size = 2048
    intermediate_size = 8192
    num_hidden_layers = 16

    max_hidden_size = 0

    try:
        while True:
            clear_gpu_memory()
            
            if local_rank == 0:
                print(f"Attempting model with hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_hidden_layers={num_hidden_layers}")
            model = create_and_initialize_model(hidden_size, intermediate_size, num_hidden_layers, ds_config)
            total_params = sum(p.numel() for p in model.parameters())
            if local_rank == 0:
                print(f"Total parameter count: {total_params:,}")

            try:
                model_engine, _, _, _ = deepspeed.initialize(
                    model=model,
                    model_parameters=model.parameters(),
                    config=ds_config
                )

                # Move tokenizer inputs to the correct device
                input_text = "DeepSpeed is"
                input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

                # If initialization succeeds, update max_hidden_size
                if local_rank == 0:
                    max_hidden_size = hidden_size
                
                # Test the model and measure tokens per second
                num_tokens = 50  # Number of tokens to generate
                start_time = time.time()
                
                with torch.no_grad():
                    output = model_engine.generate(input_ids, max_length=num_tokens, num_return_sequences=1)
                
                end_time = time.time()
                generation_time = end_time - start_time
                tokens_per_second = num_tokens / generation_time
                
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                if local_rank == 0:
                    print(f"Generated: {generated_text}")
                    print(f"Tokens per second: {tokens_per_second:.2f}")
                
                # Increase model size for next iteration
                hidden_size = int(hidden_size * 1.5)
                intermediate_size = int(intermediate_size * 1.5)
                num_hidden_layers = int(num_hidden_layers * 1.5)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if local_rank == 0:
                        print("Out of GPU memory. Stopping.")
                    break
                else:
                    raise e
            
            finally:
                del model
                if 'model_engine' in locals():
                    del model_engine

    except Exception as e:
        if local_rank == 0:
            print(f"An error occurred: {e}")

    if local_rank == 0:
        print(f"Maximum model size achieved: hidden_size={max_hidden_size}")
        print(f"Using {num_gpus} GPUs with ZeRO Stage 3 for model parallelism")

if __name__ == "__main__":
    main()
