


import torch
import time
import torch.optim as optim
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, LlamaConfig
from dataset import SubsetFineWebEdu2Loader

device = torch.device('cuda:6')
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
model.to(device)
model.train()
optimizer = optim.AdamW(
    model.parameters(),
    lr = 0.00005,  # Adjusted learning rate for 1B model
    betas = ( 0.9, 0.98 ), # Adjusted betas for 1B model
    weight_decay = 0.01  # Adjusted weight decay for 1B model
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)  # Adjusted step size for 1B model

for epoch in range(20):  # Adjusted epochs for better convergence for 1B model
    pages = SubsetFineWebEdu2Loader.next_pages(
        offset = int(time.time()),
        n_pages = 1,
        seed = 1 
    )
    dataset = SubsetFineWebEdu2Loader(
        batch_size = 8,  # Adjusted batch size for 1B model
        sequence_length = 1024,
        pages_info = pages,
        tokenizer = tokenizer
    )
    for idx, batch in enumerate( dataset ):
        optimizer.zero_grad()
        input_ids = torch.tensor(batch, dtype=torch.long).to(device)
        labels = input_ids.clone()
        labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)
        outputs = model(input_ids = input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        print ( f'Epoch: {epoch}, Batch: {idx}, Loss: {loss.item()}' )
        del input_ids, labels, outputs
        torch.cuda.empty_cache()