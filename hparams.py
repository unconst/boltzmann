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

from types import SimpleNamespace
from transformers import AutoTokenizer, LlamaConfig

def load_hparams() -> SimpleNamespace:
    hparams = {
        # Steps between full state syncing
        'epoch_length': 2_000,
        # Delta compression rate.
        'compression': 300,
        # Global sequence length
        'sequence_length': 1024,
        # AutoTokenizer name.
        'tokenizer_name': 'gpt2',
        # Size of the hidden layers in the model.
        'hidden_size': 2040,
        # Number of hidden layers in the model.
        'num_hidden_layers': 12,
        # Number of attention heads in the model.
        'num_attention_heads': 12,
        # Size of the intermediate (feed-forward) layer in the model.
        'intermediate_size': 6144
    }
    # Convert the dictionary to a SimpleNamespace
    hparams_ns = SimpleNamespace(**hparams)
    hparams_ns.tokenizer = AutoTokenizer.from_pretrained( hparams_ns.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True )
    hparams_ns.tokenizer.pad_token = hparams_ns.tokenizer.eos_token
    hparams_ns.model_config = LlamaConfig(
        vocab_size = hparams_ns.tokenizer.vocab_size,
        hidden_size = hparams_ns.hidden_size,
        num_hidden_layers = hparams_ns.num_hidden_layers,
        num_attention_heads = hparams_ns.num_attention_heads,
        intermediate_size = hparams_ns.intermediate_size,
    )
    return hparams_ns
