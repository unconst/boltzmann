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

import os
import boto3
from dotenv import dotenv_values
from transformers import AutoTokenizer
from transformers import LlamaConfig

# Number of epochs before a model becomes stale (and no longer considered active for incentive.)
EPOCH_CLIFF = 512

# Base sample probability
BASE_PROBABILITY = 1

# Skews higher weights by exponential factor with this temperature term.
TEMPERATURE = 5

# How much (out of 1) the local evaluation counts relative to global.
LOCAL_DOMINANCE = 0.5

# Moving average alpha for the validator
BASE_ALPHA = 0.0001

# Number of blocks to eval per uid.
BLOCKS_PER_EPOCH = 60

# Global sequence length
SEQUENCE_LENGTH = 4096

# Size of the local eval window.
WINDOW_SIZE = 100

# Number of new pages added to the local window every block.
WINDOW_SPEED = 4

# Number of epochs before setting weights on chain.
BLOCKS_PER_SET_WEIGHT = 100

# Instantiate the AWS S3 client.
env_config = {**dotenv_values(".env"), **os.environ}  # Load environment variables.
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')  # AWS access key ID.
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')  # AWS secret access key.
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',  # AWS region.
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Instantiate the global tokenizer.
TOKENIZER_TYPE = 'gpt4'

if TOKENIZER_TYPE == 'gpt2': 
    TOKENIZER: AutoTokenizer = AutoTokenizer.from_pretrained(
        'gpt2', verbose=False, clean_up_tokenization_spaces=True
    )
    TOKENIZER.pad_token = TOKENIZER.eos_token  # Set the padding token.
    
elif TOKENIZER_TYPE == 'gpt4':
    TOKENIZER: AutoTokenizer = AutoTokenizer.from_pretrained(
        'Xenova/gpt-4', verbose=False, clean_up_tokenization_spaces=True
    )
    TOKENIZER.pad_token = TOKENIZER.eos_token  # Set the padding token.
else:
    raise ValueError(f'No tokenizer for type: {TOKENIZER_TYPE}')


MODEL_SIZE = '7B'

if MODEL_SIZE == '1B':
    # Instantiate the global config.
    MODEL_CONFIG = LlamaConfig(
        vocab_size = TOKENIZER.vocab_size,
        hidden_size = 2040,  # Reduced hidden size to fit in memory if needed.
        num_hidden_layers = 12,
        num_attention_heads = 12,
        intermediate_size = 6144
    )

elif MODEL_SIZE == '7B':
    MODEL_CONFIG = LlamaConfig(
        vocab_size = TOKENIZER.vocab_size,
        hidden_size = 4096,
        num_hidden_layers = 32,
        num_attention_heads = 32,
        intermediate_size = 11008
    )
    
elif MODEL_SIZE == '14B':
    MODEL_CONFIG = LlamaConfig(
        vocab_size = TOKENIZER.vocab_size,
        hidden_size = 5120,
        num_hidden_layers = 55,
        num_attention_heads = 40,
        intermediate_size = 13824
    )
    
else:
    raise ValueError(f'No model size for size: {MODEL_SIZE}')
