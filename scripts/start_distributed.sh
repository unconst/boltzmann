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

# Close down all previous processes and restart them.

pm2 sendSignal SIGINT all
pm2 delete all
# Delete items from bucket
BUCKET=${1:-cont2}
PROJECT=${2:-aesop}
python3 tools/clean.py --bucket $BUCKET

# Number of GPUs to use.
NGPU=${NGPU:-1}
# The master port for distributed training.
MASTER_PORT=${MASTER_PORT:-29500}
# Which rank should be logged
LOG_RANK=${LOG_RANK:-0}
# Uncomment for debugging. 
export TORCHELASTIC_ERROR_FILE=error.log
export TORCHELASTIC_DEBUG=1
export PYTHONFAULTHANDLER=1
# Additional configuration options.
CONFIG_FILE="--job_config_file=train_configs/llama2_13b.toml"

# Start the miner using torchrun with distributed training.
pm2 start "torchrun --nproc_per_node=${NGPU} \
    --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank \
    --tee 3 miner.py ${CONFIG_FILE}" --name Miner --interpreter none
# # LLAMA 3.1
# python ./datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3-8B --tokenizer_path "original" --hf_token=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
# # LLAMA 2
# python ./datasets/download_tokenizer.py --repo_id meta-llama/Llama-2-13b-hf --hf_token=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

