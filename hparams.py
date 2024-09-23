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
import json
import requests
import os
from types import SimpleNamespace
from transformers import AutoTokenizer, LlamaConfig
import time

# Cache file path
CACHE_FILE = "hparams_cache.json"
# Cache expiration time (24 hours in seconds)
CACHE_EXPIRATION = 24 * 60 * 60


def load_from_cache() -> dict:
    """
    Load hyperparameters from the cache file if it exists and is not expired.

    Returns:
        dict: Cached hyperparameters or None if cache is invalid or expired.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cached_data = json.load(f)

        if time.time() - cached_data["timestamp"] < CACHE_EXPIRATION:
            return cached_data["hparams"]

    return None


def cache_hparams(hparams: dict):
    """
    Cache the hyperparameters to a local file.

    Args:
        hparams (dict): Hyperparameters to cache.
    """
    cache_data = {"timestamp": time.time(), "hparams": hparams}
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f)


def get_default_hparams() -> dict:
    """
    Return the default hyperparameters.

    Returns:
        dict: Default hyperparameters.
    """
    return {
        "epoch_length": 25000,
        "compression": 300,
        "sequence_length": 2048,
        "tokenizer_name": "gpt2",
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "activation_function": "swiGLU",
        "max_position_embeddings": 2048,
    }


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
    Load hyperparameters from a GitHub gist, with caching and fallback mechanisms.

    Returns:
        SimpleNamespace: A namespace containing the hyperparameters and model configuration.

    Example:
        hparams = load_hparams()
        print(hparams.hidden_size)
        print(hparams.model_config)
    """
    gist_url = "https://gist.githubusercontent.com/distributedstatemachine/75e8db446d2f1eaf1417f06d55765a32/raw/hprams.json"

    try:
        # Attempt to fetch from the gist first
        response = requests.get(gist_url, timeout=10)
        response.raise_for_status()
        hparams = json.loads(response.text)
        # Cache the new parameters
        cache_hparams(hparams)
        print("Successfully loaded parameters from gist.")
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"Error loading parameters from gist: {e}")
        print("Attempting to load from cache...")

        # Try to load from cache
        cached_hparams = load_from_cache()
        if cached_hparams:
            print("Successfully loaded parameters from cache.")
            hparams = cached_hparams
        else:
            print("Cache not available. Falling back to default parameters.")
            hparams = get_default_hparams()

    return create_namespace(hparams)
