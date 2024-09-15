]3 | _\~ ~|~ /? () 
                                        
---
BISTRO – Bittensor Incentivized and Scalable Training with Reward Optimization. 
---

# How it works

Validators on Bistro evaluate the models that miners upload to the S3 buckets attached to their keys on the chain. 
At any block model are evaluated on pages that are pulled from two sets of pages from the dataset eval and holdout.
Eval pages are random pages pulled from within a window of pages (unique to each miner) but deterministic based on the block and 
Holdout pages, random pages pulled from the full dataset. The miners must maximize their performance on their unique window of pages while still maximizing
their performance on the holdout set. The incentives are designed to force miners to train models which can be merged with each other stimulating 
inter model communication as a nessecity under the reward landscape.

# Step 1.
  - Create an S3 <Bucket> on AWS and add export your AWS API Key.
  - `export AWS_SECRET_ACCESS_KEY=`
  - `export AWS_ACCESS_KEY_ID=`

# Step 2.
  - Install python3 requirements.
  - `python3 -m pip install -r requirements.txt`

# Step 3. 
  - Register your miner on subnet 212 on testnet.
  - `btcli s register --wallet.name <> --wallet.hotkey <> --subtensor.network test --netuid 212`

# Step 4.
  - Run your miner.
  - `python3 miner.py --wallet.name <> --wallet.hotkey <> --subtensor.network test --netuid 212 --bucket <Bucket>`

# Step 5.
  - Run your validator.
  - `python3 validator.py --wallet.name <> --wallet.hotkey <> --subtensor.network test --netuid 212 --bucket <Bucket>`

