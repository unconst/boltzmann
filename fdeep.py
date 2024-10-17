import torch
import time
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
import deepspeed
import gc
import os
import random
from tqdm import tqdm
import logging
from tqdm import tqdm
from dataset import DatasetLoader

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
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        model.enable_input_require_grads()
        model.config.use_cache = False
        # Enable xFormers memory-efficient attention if available
        try:
            model.set_use_memory_efficient_attention_xformers(True)
        except AttributeError:
            pass  # Met

    return model

async def main():
    # Remove manual initialization of the distributed process group
    # DeepSpeed launcher handles this automatically
    deepspeed.init_distributed()  # Not needed when using deepspeed launcher

    # import os

    # os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
    # os.environ['LOCAL_RANK'] = str(local_rank)
    # os.environ['RANK'] = str(local_rank)  

    # Get local rank and set device
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    # Configure DeepSpeed for ZeRO Stage 3 with optimizations
    batch_size: int = 16
    ds_config: dict = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "train_batch_size": batch_size * num_gpus,
        "bf16": {"enabled": True},  # Enable bfloat16 precision
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 1e-2
            }
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": False
        },
        "logging": {"level": "ERROR"},
    }
    
    # Set backend options for performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/LLaMA-2-7B-32K", verbose=False, clean_up_tokenization_spaces=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Initial model size
    hidden_size = 4864
    intermediate_size = hidden_size * 4
    num_hidden_layers = 40

    max_hidden_size = 0
    clear_gpu_memory()
            
    if local_rank == 0:
        print(f"Attempting model with hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_hidden_layers={num_hidden_layers}")
    model = create_and_initialize_model(hidden_size, intermediate_size, num_hidden_layers, ds_config)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    if local_rank == 0:
        print(f"Total parameter count: {total_params:,}")

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        # model_parameters=model.parameters(),
        config=ds_config
    )        
    
    # Prepare inputs
    batch_size = ds_config["train_micro_batch_size_per_gpu"]
    sequence_length = 2048    
    pages = await DatasetLoader.next_pages(
        offset = 1,
        n_pages = 1,
        seed = 1
    )
    dataset = await DatasetLoader.create(
        batch_size = batch_size,
        sequence_length = sequence_length,
        pages_info = pages,
        tokenizer = tokenizer
    )
    
    start_time: float = time.time()
    tokens = 0
    num_steps = -1
    for step, batch in tqdm(enumerate(dataset)):
        if step >= num_steps and num_steps != -1:
            break

        # Convert batch to tensor and move to the correct device
        input_ids: torch.Tensor = torch.tensor(batch, dtype=torch.long).to(device).contiguous()
        # Prepare labels; ignore padding tokens for loss computation
        labels: torch.Tensor = input_ids.clone()
        labels = torch.where(labels == tokenizer.pad_token_id, -100, labels).contiguous()

        # Forward pass with labels to compute the loss
        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss: torch.Tensor = outputs.loss

        # Check if loss is valid
        # if loss is None:
        #     raise ValueError("Loss is None. Please ensure that labels are correctly set.")

        # Backward pass
        model_engine.backward(loss)

        Optimization step
        model_engine.step()

        tokens += input_ids.numel()

    elapsed_time: float = time.time() - start_time
    # Print results
    if local_rank == 0:
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Tokens processed: {tokens}")
        print(f"Tokens per second: {tokens / elapsed_time:.2f}")
            
    del model
    if 'model_engine' in locals():
        del model_engine

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
