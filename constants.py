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
EPOCH_CLIFF = 256

# Base sample probability
BASE_PROBABILITY = 1

# Skews higher weights by exponential factor with this temperature term.
TEMPERATURE = 5

# How much (out of 1) the local evaluation counts relative to global.
LOCAL_DOMINANCE = 0.33

# Moving average alpha for the validator
BASE_ALPHA = 0.0001

# Number of blocks to eval per uid.
BLOCKS_PER_EPOCH = 20

# Global sequence length
SEQUENCE_LENGTH = 2048

# Size of the local eval window.
WINDOW_SIZE = 100

# Number of new pages added to the local window every block.
WINDOW_SPEED = 10

# Number of epochs before setting weights on chain.
EPOCHS_PER_SET_WEIGHTS = 1

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
TOKENIZER: AutoTokenizer = AutoTokenizer.from_pretrained(
    'gpt2', verbose=False, clean_up_tokenization_spaces=True
)
TOKENIZER.pad_token = TOKENIZER.eos_token  # Set the padding token.

# Instantiate the global config.
MODEL_CONFIG = LlamaConfig(
    vocab_size = TOKENIZER.vocab_size,
    hidden_size = 2040,  # Reduced hidden size to fit in memory if needed.
    num_hidden_layers = 12,
    num_attention_heads = 12,
    intermediate_size = 6144
)