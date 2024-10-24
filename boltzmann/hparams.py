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
import json
import time
import requests
from types import SimpleNamespace
from transformers import AutoTokenizer, LlamaConfig

from common import *

# Cache file path
HPARAMS_FILE = "hparams.json"

def create_namespace(hparams: dict) -> SimpleNamespace:
    """
    Create a SimpleNamespace from the hyperparameters and add model configuration.

    Args:
        hparams (dict): Hyperparameters dictionary.

    Returns:
        SimpleNamespace: Namespace containing hyperparameters and model configuration.
    """
    hparams_ns = SimpleNamespace(**hparams)

    hparams_ns.tokenizer = AutoTokenizer.from_pretrained(
        hparams_ns.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True
    )
    hparams_ns.tokenizer.pad_token = hparams_ns.tokenizer.eos_token

    hparams_ns.model_config = LlamaConfig(
        vocab_size=hparams_ns.tokenizer.vocab_size,
        hidden_size=hparams_ns.hidden_size,
        num_hidden_layers=hparams_ns.num_hidden_layers,
        num_attention_heads=hparams_ns.num_attention_heads,
        intermediate_size=hparams_ns.intermediate_size,
        num_key_value_heads=hparams_ns.num_key_value_heads,
        activation_function=hparams_ns.activation_function,
        max_position_embeddings=hparams_ns.max_position_embeddings,
    )

    return hparams_ns

def load_hparams() -> SimpleNamespace:
    """
    Load hyperparameters from a GitHub file, with caching and fallback mechanisms.

    Returns:
        SimpleNamespace: A namespace containing the hyperparameters and model configuration.

    Example:
        hparams = load_hparams()
        print(hparams.hidden_size)
        print(hparams.model_config)
    """
    github_url = f"https://raw.githubusercontent.com/unconst/cont/master/hparams.json?timestamp={int(time.time())}"
    try:
        # Attempt to fetch from the GitHub file first
        response = requests.get(github_url, timeout=10, headers={'Cache-Control': 'no-cache'})
        response.raise_for_status()
        hparams = json.loads(response.text)
        logger.debug("Successfully loaded parameters from GitHub.")
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.debug(f"Error loading parameters from GitHub: {e}")
        logger.debug("Attempting to load from cache...")
        with open(HPARAMS_FILE, "r") as f:
            hparams = json.load(f)
    # Cache the new parameters
    with open(HPARAMS_FILE, "w") as f:
        json.dump(hparams, f, indent=4)
    return create_namespace(hparams)
