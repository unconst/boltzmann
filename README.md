]3 | _\~ ~|~ /? () 
                                        
---
BISTRO – Bittensor Incentivized and Scalable Training with Reward Optimization. 
---

# Mechanism

Validation design explained in pseudo-code:
```markdown
# Get the master model state. 
master = get_master()

# Iterate over each miner uid.
for uid in metagraph.uids

  # Add the miner's delta to the master model multiplied by incentive.
  add_delta( master, get_delta( uid ) * metagraph.I[ uid ] )

# Iterate over each miner uid.
for uid in metagraph.uids:

  # Eval the master on the miner's current local pages.
  local_pages = get_pages( subtensor.block, uid )
  
  # Eval the delta on the master.
  loss_with_delta = eval( local_pages, master )

  # Remove the miner's delta from the master.
  remove_delta( master, get_delta( uid ) * metagraph.I[ uid ] )
  
  # Eval the master without the delta.
  loss_without_delta = eval( local_pages, master )
  
  # Score the miner based on the difference.
  scores[ uid ] = loss_with_delta - loss_without_delta

  # Add the miner's delta back to the master model.
  add_delta( master, get_delta( uid ) * metagraph.I[ uid ] )

# Softmax scores.
weights = softmax( weights * temperature )

# Set weights on chain
subtensor.set_weights( weights )
```

Miner are rewarded for producing a compressed delta on the current master model which reduces the loss maximally when added in conjunction with the deltas produced by other miners in the network weighted by incentive. Mathematically this can be expressed as :

The master model's parameters after applying the miners' deltas are updated as:

\[
\theta_{\text{master}} = \theta_{\text{initial}} + \sum_{i} w_i I_i \Delta_i
\]

where the weight \( w_i \) for each miner is determined by the softmax of the loss change:

\[
w_i = \frac{\exp\left(\frac{\Delta L_i}{T}\right)}{\sum_j \exp\left(\frac{\Delta L_j}{T}\right)}, \quad \Delta L_i = L(\theta_{\text{master}}^{+i}, \mathcal{D}_i) - L(\theta_{\text{master}}^{-i}, \mathcal{D}_i)
\]

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

