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

DEBUG=${1:-false}
PROJECT=${2:-aesop}

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

pdone() {
    printf "  ${tty_green}[✔]${tty_bold} %s${tty_reset}\n" "$*"
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

getc() {
  local save_state
  save_state="$(/bin/stty -g)"
  /bin/stty raw -echo
  IFS='' read -r -n 1 -d '' "$@"
  /bin/stty "${save_state}"
}

wait_for_user() {
  local c
  echo
  echo "Press ${tty_bold}RETURN${tty_reset}/${tty_bold}ENTER${tty_reset} to continue or any other key to abort:"
  getc c
  # we test for \r and \n because some stuff does \r instead
  if ! [[ "${c}" == $'\r' || "${c}" == $'\n' ]]
  then
    exit 1
  fi
}

execute() {
    ohai "Running: $*"
    if ! "$@"; then
        abort "Failed during: $*"
    fi
}

have_sudo_access() {
    if [ "$EUID" -ne 0 ]; then
        if ! command -v sudo &> /dev/null; then
            warn "sudo command not found. Please install sudo or run as root."
            return 1
        fi
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
        warn "Sudo access is required, attempting to run without sudo"
        ohai "$*"
        if ! "$@"; then
            abort "Failed to execute: $*"
        fi
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

clear
echo ""
echo ""
echo " ______   _____         _______ ______ _______ _______ __   _ __   _"
echo " |_____] |     | |         |     ____/ |  |  | |_____| | \  | | \  |"
echo " |_____] |_____| |_____    |    /_____ |  |  | |     | |  \_| |  \_|"
echo "                                                                    "
echo ""
echo ""

wait_for_user

# Install Git if not present
ohai "Installing requirements ..."
if ! command -v git &> /dev/null; then
    ohai "Installing git ..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ohai "Detected Linux"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" == "ubuntu" || "$ID_LIKE" == *"ubuntu"* ]]; then
                ohai "Detected Ubuntu, installing Git..."
                if [[ "$DEBUG" == "true" ]]; then
                    execute_sudo apt-get update -y
                    execute_sudo apt-get install git -y
                else
                    execute_sudo apt-get update -y > /dev/null 2>&1
                    execute_sudo apt-get install git -y > /dev/null 2>&1
                fi
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
        if [[ "$DEBUG" == "true" ]]; then
            execute brew install git
        else
            execute brew install git > /dev/null 2>&1
        fi
    else
        abort "Unsupported OS type: $OSTYPE"
    fi
else
    pdone "Installed Git"
fi


# Check if npm is installed
if ! command -v npm &> /dev/null; then
    ohai "Installing npm ..."
    if ! command -v node &> /dev/null; then
        ohai "Node.js could not be found, installing..."
        if ! curl -fsSL https://deb.nodesource.com/setup_14.x | execute_sudo -E bash -; then
            abort "Failed to download Node.js setup script"
        fi
        if ! execute_sudo apt-get install -y nodejs; then
            abort "Failed to install Node.js"
        fi
    fi
    if ! curl -L https://www.npmjs.com/install.sh | sh; then
        abort "Failed to install npm"
    fi
fi
pdone "Installed npm"


# Install pm2
if ! command -v pm2 &> /dev/null; then
    ohai "Installing pm2 ..."
    if [[ "$DEBUG" == "true" ]]; then
        execute npm install pm2 -g
    else
        execute npm install pm2 -g > /dev/null 2>&1
    fi
fi
pdone "Installed pm2"

# Install Python 3.12 if not installed
if ! command -v python3.12 &> /dev/null; then
    ohai "Installing python3.12 ..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ohai "Detected Linux"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" == "ubuntu" || "$ID_LIKE" == *"ubuntu"* ]]; then
                ohai "Detected Ubuntu, installing Python 3.12..."
                if [[ "$DEBUG" == "true" ]]; then
                    execute_sudo add-apt-repository ppa:deadsnakes/ppa -y
                    execute_sudo apt-get update -y
                    execute_sudo apt-get install python3.12 -y
                else
                    execute_sudo add-apt-repository ppa:deadsnakes/ppa -y > /dev/null 2>&1
                    execute_sudo apt-get update -y > /dev/null 2>&1
                    execute_sudo apt-get install python3.12 -y > /dev/null 2>&1
                fi
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
        if [[ "$DEBUG" == "true" ]]; then
            execute brew install python@3.12
        else
            execute brew install python@3.12 > /dev/null 2>&1
        fi
    else
        abort "Unsupported OS type: $OSTYPE"
    fi
fi
pdone "Installed python3.12"

touch ~/.bash_profile

# Prompt the user for AWS credentials and inject them into the bash_profile file if not already stored
ohai "Getting AWS credentials ..."
if ! grep -q "AWS_ACCESS_KEY_ID" ~/.bash_profile || ! grep -q "AWS_SECRET_ACCESS_KEY" ~/.bash_profile || ! grep -q "BUCKET" ~/.bash_profile; then
    clear
    warn "This script will store your AWS credentials in your ~/.bash_profile file."
    warn "This is not secure and is not recommended."
    read -p "Do you want to proceed? [y/N]: " proceed
    if [[ "$proceed" != "y" && "$proceed" != "Y" ]]; then
        abort "Aborted by user."
    fi

    read -p "Enter your AWS Access Key ID: " AWS_ACCESS_KEY_ID
    read -p "Enter your AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
    read -p "Enter your S3 Bucket Name: " BUCKET

    echo "export AWS_ACCESS_KEY_ID=\"$AWS_ACCESS_KEY_ID\"" >> ~/.bash_profile
    echo "export AWS_SECRET_ACCESS_KEY=\"$AWS_SECRET_ACCESS_KEY\"" >> ~/.bash_profile
    echo "export BUCKET=\"$BUCKET\"" >> ~/.bash_profile
fi

# Source the bashrc file to apply the changes
source ~/.bash_profile
pdone "Found AWS credentials"

ohai "Installing Boltzmann ..."
# Check if we are inside the boltzmann repository
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    REPO_PATH="."
else
    if [ ! -d "boltzmann" ]; then
        ohai "Cloning boltzmann ..."
        execute git clone https://github.com/unconst/boltzmann
        REPO_PATH="boltzmann/"
    else
        REPO_PATH="boltzmann/"
    fi
fi
pdone "Pulled Boltzmann $REPO_PATH"

# Create a virtual environment if it does not exist
if [ ! -d "$REPO_PATH/venv" ]; then
    ohai "Creating virtual environment at $REPO_PATH..."
    if [[ "$DEBUG" == "true" ]]; then
        execute python3.12 -m venv "$REPO_PATH/venv"
    else
        execute python3.12 -m venv "$REPO_PATH/venv" > /dev/null 2>&1
    fi
fi
pdone "Created venv at $REPO_PATH"


pdone "Installing python requirements.txt"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source $REPO_PATH/venv/bin/activate > /dev/null 2>&1
fi
pdone "Activated venv at $REPO_PATH"

if [[ "$DEBUG" == "true" ]]; then
    execute pip install -r $REPO_PATH/requirements.txt
    execute pip install --upgrade cryptography pyOpenSSL
else
    execute pip install -r $REPO_PATH/requirements.txt > /dev/null 2>&1
    execute pip install --upgrade cryptography pyOpenSSL > /dev/null 2>&1
fi
pdone "Installed requirements"

# Check for GPUs
ohai "Checking for GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    warn "nvidia-smi command not found. Please ensure NVIDIA drivers are installed."
    NUM_GPUS=0
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    if [ "$NUM_GPUS" -gt 0 ]; then
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | while read -r memory; do
            pdone " - Found GPU with $((memory / 1024)) GB"
        done
    else
        warn "No GPUs found on this machine."
    fi
fi

# Check system RAM
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
else
    warn "Cannot determine system RAM. 'free' command not found."
fi

# Create the default key
ohai "Creating wallets ..."
if ! python3 -c "import bittensor as bt; w = bt.wallet(); print(w.coldkey_file.exists_on_device())" | grep -q "True"; then
    execute btcli w new_coldkey --wallet.path ~/.bittensor/wallets --wallet.name default --n-words 12 
fi
pdone "Attained Wallet(default)"

# Ensure btcli is installed
if ! command -v btcli &> /dev/null; then
    abort "btcli command not found. Please ensure it is installed."
fi

# Create hotkeys and register them
if [ "$NUM_GPUS" -gt 0 ]; then
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        # Check if the hotkey file exists on the device
        exists_on_device=$(python3 -c "import bittensor as bt; w = bt.wallet(hotkey='C$i'); print(w.hotkey_file.exists_on_device())" 2>/dev/null)
        if [ "$exists_on_device" != "True" ]; then
            echo "n" | btcli wallet new_hotkey --wallet.name default --wallet.hotkey C$i --n-words 12 > /dev/null 2>&1;
        fi
        pdone "Created Hotkey( C$i )"

        # Check if the hotkey is registered on subnet 220
        is_registered=$(python3 -c "import bittensor as bt; w = bt.wallet(hotkey='C$i'); sub = bt.subtensor('test'); print(sub.is_hotkey_registered_on_subnet(hotkey_ss58=w.hotkey.ss58_address, netuid=220))" 2>/dev/null)
        if [[ "$is_registered" != *"True"* ]]; then
            ohai "Registering key on subnet 220"
            btcli subnet pow_register --wallet.name default --wallet.hotkey C$i --netuid 220 --subtensor.network test --no_prompt > /dev/null 2>&1;
        fi
        pdone "Registered Hotkey( C$i )"
    done
else
    warn "No GPUs found. Skipping hotkey creation."
    exit
fi
pdone "Registered $NUM_GPUS keys to subnet 220"

ohai "Logging into wandb..."
if [[ "$DEBUG" == "true" ]]; then
    execute wandb login
else
    execute wandb login > /dev/null 2>&1
fi
pdone "Initialized wandb"


# Delete items from bucket

ohai "Cleaning bucket $BUCKET..."
if [[ "$DEBUG" == "true" ]]; then
    execute python3 $REPO_PATH/tools/clean.py --bucket "$BUCKET"
else
    execute python3 $REPO_PATH/tools/clean.py --bucket "$BUCKET" > /dev/null 2>&1
fi
pdone "Cleaned bucket"

# Close down all previous processes and restart them
if pm2 list | grep -q 'online'; then
    ohai "Stopping old pm2 processes..."
    pm2 delete all
    pdone "Stopped old processes"
fi

# Start all the processes again
if [ "$NUM_GPUS" -gt 0 ]; then
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        # Adjust GPU index for zero-based numbering
        GPU_INDEX=$i
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n "$((i + 1))p")
        if [ -z "$GPU_MEMORY" ]; then
            warn "Could not get GPU memory for GPU $i"
            continue
        fi
        # Determine batch size based on GPU memory
        # This section adjusts the batch size for the miner based on the available GPU memory
        # Higher memory allows for larger batch sizes, which can improve performance
        if [ "$GPU_MEMORY" -ge 80000 ]; then
            # For GPUs with 80GB or more memory, use a batch size of 6
            BATCH_SIZE=6
        elif [ "$GPU_MEMORY" -ge 40000 ]; then
            # For GPUs with 40GB to 79GB memory, use a batch size of 3
            BATCH_SIZE=3
        elif [ "$GPU_MEMORY" -ge 20000 ]; then
            # For GPUs with 20GB to 39GB memory, use a batch size of 1
            BATCH_SIZE=1
        else
            # For GPUs with less than 20GB memory, also use a batch size of 1
            # This ensures that even lower-end GPUs can still participate
            BATCH_SIZE=1
        fi
        ohai "Starting miner on GPU $GPU_INDEX with batch size $BATCH_SIZE..."
        if [[ "$DEBUG" == "true" ]]; then
            execute pm2 start "$REPO_PATH/miner.py" --interpreter python3 --name C$i -- --actual_batch_size "$BATCH_SIZE" --wallet.name default --wallet.hotkey C$i --bucket "$BUCKET" --device cuda:$GPU_INDEX --use_wandb --project "$PROJECT"
        else
            execute pm2 start "$REPO_PATH/miner.py" --interpreter python3 --name C$i -- --actual_batch_size "$BATCH_SIZE" --wallet.name default --wallet.hotkey C$i --bucket "$BUCKET" --device cuda:$GPU_INDEX --use_wandb --project "$PROJECT" > /dev/null 2>&1
        fi
    done
else
    warn "No GPUs found. Skipping miner startup."
fi
pdone "Started miners"
pm2 list

echo ""
pdone "SUCCESS"
echo ""
