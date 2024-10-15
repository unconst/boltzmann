#!/usr/bin/env bash

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

set -euo pipefail

trap 'abort "An unexpected error occurred."' ERR

# Set up colors and styles
if [[ -t 1 ]]; then
    tty_escape() { printf "\033[%sm" "$1"; }
else
    tty_escape() { :; }
fi
tty_mkbold() { tty_escape "1;$1"; }
tty_blue="$(tty_mkbold 34)"
tty_red="$(tty_mkbold 31)"
tty_green="$(tty_mkbold 32)"
tty_yellow="$(tty_mkbold 33)"
tty_bold="$(tty_mkbold 39)"
tty_reset="$(tty_escape 0)"

ohai() {
    printf "${tty_blue}==>${tty_bold} %s${tty_reset}\n" "$*"
}

info() {
    printf "${tty_green}%s${tty_reset}\n" "$*"
}

warn() {
    printf "${tty_yellow}Warning${tty_reset}: %s\n" "$*" >&2
}

error() {
    printf "${tty_red}Error${tty_reset}: %s\n" "$*" >&2
}

abort() {
    error "$@"
    exit 1
}

execute() {
    ohai "Running: $*"
    if ! "$@"; then
        abort "Failed during: $*"
    fi
}

have_sudo_access() {
    if [ "$EUID" -ne 0 ]; then
        if ! sudo -n true 2>/dev/null; then
            warn "This script requires sudo access to install packages. Please run as root or ensure your user has sudo privileges."
            return 1
        fi
    fi
    return 0
}

execute_sudo() {
    if have_sudo_access; then
        ohai "sudo $*"
        if ! sudo "$@"; then
            abort "Failed to execute: sudo $*"
        fi
    else
        abort "Sudo access is required"
    fi
}

test_curl() {
  if [[ ! -x "$1" ]]
  then
    return 1
  fi

  local curl_version_output curl_name_and_version
  curl_version_output="$("$1" --version 2>/dev/null)"
  curl_name_and_version="${curl_version_output%% (*}"
  version_ge "$(major_minor "${curl_name_and_version##* }")" "$(major_minor "${REQUIRED_CURL_VERSION}")"
}

# Install Git if not present
if ! command -v git &> /dev/null; then
    ohai "Git could not be found, installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ohai "Detected Linux"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" == "ubuntu" || "$ID_LIKE" == *"ubuntu"* ]]; then
                ohai "Detected Ubuntu, installing Git..."
                execute_sudo apt-get update -y > /dev/null 2>&1
                execute_sudo apt-get install git -y > /dev/null 2>&1
            else
                warn "Unsupported Linux distribution: $ID"
                abort "Cannot install Git automatically"
            fi
        else
            warn "Cannot detect Linux distribution"
            abort "Cannot install Git automatically"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        ohai "Detected macOS, installing Git..."
        if ! command -v brew &> /dev/null; then
            warn "Homebrew is not installed, installing Homebrew..."
            execute /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        execute brew install git > /dev/null 2>&1
    else
        abort "Unsupported OS type: $OSTYPE"
    fi
else
    info "Git is already installed"
fi

# Clone the repository if not present
if [ ! -d "cont" ]; then
    ohai "Cloning the repository..."
    execute git clone https://github.com/unconst/cont
else
    info "Repository already cloned"
fi


# Check if npm is installed
if ! command -v npm &> /dev/null; then
    ohai "npm could not be found, installing..."
    if ! command -v node &> /dev/null; then
        ohai "Node.js could not be found, installing..."
        if ! curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash -; then
            abort "Failed to download Node.js setup script"
        fi
        if ! sudo apt-get install -y nodejs; then
            abort "Failed to install Node.js"
        fi
    fi
    if ! curl -L https://www.npmjs.com/install.sh | sh; then
        abort "Failed to install npm"
    fi
else
    info "npm is already installed"
fi

# Install pm2
if ! command -v pm2 &> /dev/null; then
    ohai "pm2 could not be found, installing..."
    execute npm install pm2 -g > /dev/null 2>&1
else
    info "pm2 is already installed"
fi

# Install Python 3.12 if not installed
if ! command -v python3.12 &> /dev/null; then
    ohai "Python 3.12 not found, installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ohai "Detected Linux"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" == "ubuntu" || "$ID_LIKE" == *"ubuntu"* ]]; then
                ohai "Detected Ubuntu, installing Python 3.12..."
                execute_sudo add-apt-repository ppa:deadsnakes/ppa -y > /dev/null 2>&1
                execute_sudo apt-get update -y > /dev/null 2>&1
                execute_sudo apt-get install python3.12 -y > /dev/null 2>&1
            else
                warn "Unsupported Linux distribution: $ID"
                abort "Cannot install Python 3.12 automatically"
            fi
        else
            warn "Cannot detect Linux distribution"
            abort "Cannot install Python 3.12 automatically"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        ohai "Detected macOS, installing Python 3.12..."
        if ! command -v brew &> /dev/null; then
            warn "Homebrew is not installed, installing Homebrew..."
            execute /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        execute brew install python@3.12 > /dev/null 2>&1
    else
        abort "Unsupported OS type: $OSTYPE"
    fi
else
    info "Python 3.12 is already installed"
fi

# Prompt the user for AWS credentials and inject them into the bashrc file if not already stored
if ! grep -q "AWS_ACCESS_KEY_ID" ~/.bashrc || ! grep -q "AWS_SECRET_ACCESS_KEY" ~/.bashrc || ! grep -q "BUCKET" ~/.bashrc; then
    warn "This script will store your AWS credentials in your ~/.bashrc file."
    warn "This is not secure and is not recommended."
    read -p "Do you want to proceed? [y/N]: " proceed
    if [[ "$proceed" != "y" && "$proceed" != "Y" ]]; then
        abort "Aborted by user."
    fi

    read -p "Enter your AWS Access Key ID: " AWS_ACCESS_KEY_ID
    read -p "Enter your AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
    read -p "Enter your S3 Bucket Name: " BUCKET

    echo "export AWS_ACCESS_KEY_ID=\"$AWS_ACCESS_KEY_ID\"" >> ~/.bashrc
    echo "export AWS_SECRET_ACCESS_KEY=\"$AWS_SECRET_ACCESS_KEY\"" >> ~/.bashrc
    echo "export BUCKET=\"$BUCKET\"" >> ~/.bashrc
    export BUCKET="$BUCKET"
else
    info "AWS credentials are already stored in ~/.bashrc"
fi

# Source the bashrc file to apply the changes
# source ~/.bashrc

# Create a virtual environment if it does not exist
if [ ! -d "venv" ]; then
    ohai "Creating virtual environment..."
    execute python3.12 -m venv venv > /dev/null 2>&1
else
    info "Virtual environment already exists"
fi

ohai "Activating virtual environment..."
source venv/bin/activate > /dev/null 2>&1

ohai "Installing requirements..."
execute pip install -r requirements.txt > /dev/null 2>&1

# Check for GPUs
ohai "Checking for GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    warn "nvidia-smi command not found. Please ensure NVIDIA drivers are installed."
    NUM_GPUS=0
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    info "Number of GPUs: $NUM_GPUS"

    if [ "$NUM_GPUS" -gt 0 ]; then
        ohai "GPU Memory Information:"
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | while read -r memory; do
            info "$((memory / 1024)) GB"
        done
    else
        warn "No GPUs found on this machine."
    fi
fi

# Check system RAM
ohai "Checking system RAM..."
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    info "Total System RAM: $TOTAL_RAM GB"
else
    warn "Cannot determine system RAM. 'free' command not found."
fi

# Create the default key
ohai "Creating the coldkey"
if ! python3 -c "import bittensor as bt; w = bt.wallet(); print(w.coldkey_file.exists_on_device())" | grep -q "True"; then
    execute btcli w new_coldkey --wallet.path ~/.bittensor/wallets --wallet.name default --n-words 12
else
    info "Default key already exists on device."
fi


# Ensure btcli is installed
if ! command -v btcli &> /dev/null; then
    abort "btcli command not found. Please ensure it is installed."
fi

# Create hotkeys and register them
if [ "$NUM_GPUS" -gt 0 ]; then
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        ohai "Creating hotkeys C$i for default..."
        echo "n" | execute btcli wallet new_hotkey --wallet.name default --wallet.hotkey C$i --n-words 12 > /dev/null 2>&1
        execute btcli subnet pow_register --wallet.name default --wallet.hotkey C$i --netuid 220 --subtensor.network test --no_prompt > /dev/null 2>&1
    done
else
    warn "No GPUs found. Skipping hotkey creation."
fi

# Login to wandb
if ! command -v wandb &> /dev/null; then
    abort "wandb command not found. Please ensure it is installed."
fi

ohai "Logging into wandb..."
execute wandb login > /dev/null 2>&1

# # Close down all previous processes and restart them
# ohai "Stopping all pm2 processes..."
# pm2 delete all

# Delete items from bucket
PROJECT=${2:-aesop}
ohai "Cleaning bucket $BUCKET..."
execute python3 tools/clean.py --bucket "$BUCKET" > /dev/null 2>&1

# Start all the processes again
if [ "$NUM_GPUS" -gt 0 ]; then
    for i in $(seq 1 $NUM_GPUS); do
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n "${i}p")
        if [ -z "$GPU_MEMORY" ]; then
            warn "Could not get GPU memory for GPU $i"
            continue
        fi
        if [ "$GPU_MEMORY" -ge 80000 ]; then
            BATCH_SIZE=6
        elif [ "$GPU_MEMORY" -ge 40000 ]; then
            BATCH_SIZE=3
        elif [ "$GPU_MEMORY" -ge 20000 ]; then
            BATCH_SIZE=2
        else
            BATCH_SIZE=1
        fi
        ohai "Starting miner on GPU $((i-1)) with batch size $BATCH_SIZE..."
        execute pm2 start miner.py --interpreter python3 --name C$i -- --actual_batch_size "$BATCH_SIZE" --wallet.name default --wallet.hotkey C$i --bucket "$BUCKET" --device cuda:$((i-1)) --use_wandb --project "$PROJECT"
    done
else
    warn "No GPUs found. Skipping miner startup."
fi

pm2 list

ohai "Script completed successfully."
