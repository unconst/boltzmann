]3 | _\~ ~|~ /? () 
                                        
---
BISTRO – Bittensor Incentivized and Scalable Training with Reward Optimization. 
---

Miners are rewarded for training the network model on the page sequences determined by the loader in dataset.py.
Miners upload slices of their models using a distro-style distributed training regime with 300x compression as default.
We use S3 buckets as an intermediary for the distribution of the model slices which makes transfer incredibly fast and scalable. 
The valdiators measure the loss difference between the model with the slices applied and with the slices removed to ensure performance from miners.
This difference becomes the raw scores which validators submit to the chain after a softmax.
There are various corner cases remaining to make this work but generally the structure is here. Good luck!

NOTE: you need a fairly large machine (H100 recommended) to run either the validator or miner.

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
  - `python3 validator.py --wallet.name <> --wallet.hotkey <> --subtensor.network test --netuid 212 --bucket <Bucket> --device <>`


```
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
```
