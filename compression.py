# The MIT License (MIT)
# Copyright © 2024 Chakana.tech

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

import torch

# Function to compress gradients
def topk_compress_gradients( grad_dict, k ):
    compressed_dict = {}
    
    # Compress each tensor in the state dict
    for key, grad in grad_dict.items():
        if grad is not None:
            # Flatten gradient
            flattened_grad = grad.flatten()
            
            # Get the top-k indices by absolute value
            topk_values, topk_indices = torch.topk(flattened_grad.abs(), k)
            
            # Save original values at these indices
            topk_original_values = flattened_grad[topk_indices]
            
            # Store compressed info
            compressed_dict[key] = {'indices': topk_indices, 'values': topk_original_values}
    
    return compressed_dict

# Function to decompress gradients
def topk_decompress_gradients( compressed_dict, original_shape_dict ):
    decompressed_dict = {}
    
    # Decompress each tensor
    for key, compressed_data in compressed_dict.items():
        # Create a zero tensor with the original shape
        original_shape = original_shape_dict[key]
        decompressed_grad = torch.zeros(original_shape).flatten()  # Create a flat tensor
        
        # Get the indices and values
        indices = compressed_data['indices']
        values = compressed_data['values']
        
        # Assign the values to the respective indices
        decompressed_grad[indices] = values
        
        # Reshape the gradient back to original shape
        decompressed_dict[key] = decompressed_grad.view(original_shape)
    
    return decompressed_dict