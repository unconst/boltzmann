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

# Creates Alice wallets with Alice, Bob, Dave, Charlie, Eve and Ferdie hotkeys.
# echo " Create wallets if not existent"
# for name in Alice Bob Charlie Dave Eve Ferdie
# do
# echo "import bittensor as bt
# w = bt.wallet(name='Alice', hotkey='$name')
# if not w.coldkey_file.exists_on_device():
#     w.create_coldkey_from_uri('//Alice', overwrite=True, use_password=False, suppress=True)
# if not w.hotkey_file.exists_on_device():
#     w.create_coldkey_from_uri('/$name', overwrite=True, use_password=False, suppress=False)
# " > create_wallet.py
# python3 create_wallet.py
# rm create_wallet.py
# done

# Close down all previous processes and restart them.
pm2 sendSignal SIGINT all
pm2 delete all

# Delete items from bucket
BUCKET=${1:-decis}
python3 tools/clean.py --bucket $BUCKET

# Start all the processes again.
pm2 start validator.py --interpreter python3 --name V1 -- --wallet.name Alice --wallet.hotkey default --bucket $BUCKET --device cuda:0 --use_wandb
pm2 start miner.py --interpreter python3 --name M1 -- --wallet.name Alice --wallet.hotkey M1 --bucket $BUCKET --device cuda:1 --use_wandb 
pm2 start miner.py --interpreter python3 --name M2 -- --wallet.name Alice --wallet.hotkey M2 --bucket $BUCKET --device cuda:2 --use_wandb
pm2 start miner.py --interpreter python3 --name M3 -- --wallet.name Alice --wallet.hotkey M3 --bucket $BUCKET --device cuda:3 --use_wandb
pm2 start miner.py --interpreter python3 --name M4 -- --wallet.name Alice --wallet.hotkey M4 --bucket $BUCKET --device cuda:5 --use_wandb
pm2 start miner.py --interpreter python3 --name M5 -- --wallet.name Alice --wallet.hotkey M5 --bucket $BUCKET --device cuda:6 --use_wandb



