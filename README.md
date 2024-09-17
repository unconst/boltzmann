]3 | _\~ ~|~ /? () 
                                        
---
BISTRO – Bittensor Incentivized and Scalable Training with Reward Optimization. 
---

# Mechanism

Validation design explained in pseudo-code:
```markdown

# Init the model
master = init_model()

# Upload the model to S3
upload( master )

# Loop forever.
loop:

  # Select a random uid to sample.
  uid = random.choice( metagraph.uids )

  # Get the latest delta uploaded by the miner. 
  delta = get_delta( uid )

  # Add the delta to the master model
  add_delta( master, delta )

  # Eval the master on the miner's current local pages.
  local_pages = get_pages( subtensor.block, uid )
  local_loss = eval( local_pages, master )

  # Eval the master on random pages from global.
  global_pages = get_random_pages()
  global_loss = eval( global_pages, master )

  # Set weights on chain
  weights[ uid ] = (1/2) * moving_average( -local_loss ) + (1/2) * moving_average( -global_loss )

  # Softmax weights: smaller negative losses getting higher scores.
  weights = softmax( weights * temperature )

  # Set weights on chain
  subtensor.set_weights( weights )

  # Check if moving average produces a reasonable update.
  if global_loss < min_loss:
      # Update min loss.
      min_loss = global_loss

      # Upload the new master (gossip the UID so miners can pull it)
      upload( master, delta = uid ) 
  
  # Otherwise, loop.
  else:
      remove_delta( master, delta )
```

Validators on Bistro evaluate gradients (or model deltas) that miners submit to intermediate S3 buckets. At any block miner deltas are evaluated 'local' and 'global' pages pulled from the dataset. Local pages are randomly sampled pages pulled from within a window of pages (unique to each miner) but deterministic based on the block, global pages are pulled at random from the full dataset. Miners must maximize their performance on their unique window of pages while regulating their performance on the global. The incentives are designed to force miners to train mergable models and to make them available in a high bandwidth environment. As the training progresses the master model updates as deltas show case reduced losses.

# Step 1.
  - Create an S3 <Bucket> on AWS and add export your AWS API Key.
  - Make sure to set the most permissive access to your bucket.
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

