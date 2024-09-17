]3 | _\~ ~|~ /? () 
                                        
---
BISTRO – Bittensor Incentivized and Scalable Training with Reward Optimization. 
---

# Mechanism

```markdown

# Init the model
master = init_model()

# Upload the model to S3
upload( master )

# Loop forever.
loop:

  # miner uids on the network.
  for uid in uids:

      # Get the latest delta uploaded by the miner. 
      delta = get_delta()

      # Add the delta to the master model
      add_delta( master, delta )

      # Eval the master on the miner's current local pages
      local_pages = get_pages( block, uid)
      local_loss = eval( local_pages, master )

      # Eval the master on random pages
      global_pages = get_random_pages()
      global_loss = eval( global_pages, master )

      # Check if moving average produces reasonable update.
      if moving_average( global_loss, uid ) > 


while True:

  metagraph = metagraph.sync()
  uid = random_uid( metagraph )
  delta = downdload( uid )



  for uid in uids:



Validators on Bistro evaluate gradients (or model deltas) that miners submit to intermediate S3 buckets.
At any block, deltas are evaluated on two sets of pages that are pulled from RefinedWeb: local and global.
Local pages are randomly sampled pulled from within a window of pages (unique to each miner) but deterministic based on the block and 
global pages are pulled at random from the full dataset. The miners must maximize their performance on their unique window of pages while regulating their performance
on the global. The incentives are designed to force miners to train mergable models and to make them available in a high bandwidth environment.

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
  - `python3 miner.py --wallet.name <> --wallet.hotkey <> --subtensor.network test --netuid 212 --bucket <Bucket> --device <>`

# Step 5.
  - Run your validator.
  - `python3 validator.py --wallet.name <> --wallet.hotkey <> --subtensor.network test --netuid 212 --bucket <Bucket> --device`

