<p align="center">
    <img src="https://raw.githubusercontent.com/unconst/bistro/main/assets/logo.png" alt="BISTRO Logo" width="200">
</p>

---

# BISTRO: Bittensor Incentivized Scalable Training with Reward Optimization

---

Welcome to the **BISTRO** project! This repository implements an innovative incentive system for decentralized machine learning training on the Bittensor network.

## Overview

**BISTRO** introduces a decentralized framework where **miners** collaboratively train a shared model by processing specific subsets of data, and **validators** ensure the quality and integrity of their contributions. The incentive mechanism is designed to reward miners for effectively training on their designated data subsets, promoting efficient and collaborative model improvement in a trustless environment.

## How It Works

### Miners

- **Model Synchronization**: Miners start by downloading the latest model state, which is a subset of model parameters (called **slices**) aggregated from other miners.
- **Training**: They receive a designated subset of data (pages) from the dataset for each window (a fixed number of blocks). They train the model on this subset, performing gradient updates.
- **Uploading Deltas**: After training, miners compute the **delta** (the change in their model parameters) and upload this delta to an S3 bucket associated with their identity.
- **Window Progression**: Miners proceed to the next window, repeating the process with new data subsets.

### Validators

- **Model Synchronization**: Validators synchronize their model state by downloading the latest aggregated slices and applying miners' deltas from the previous window.
- **Fetching Deltas**: Validators download the deltas uploaded by miners corresponding to the last window.
- **Evaluation**: Validators evaluate the miners' contributions by comparing the miners' deltas to the gradients computed locally on the same data subsets.
- **Scoring**: The reward for each miner is calculated based on the **cosine similarity** between the miner's delta and the validator's locally computed gradient. This incentivizes miners to provide genuine, high-quality updates that improve the model.

## Incentive Mechanism Explained

The incentive mechanism in **BISTRO** ensures that miners are rewarded for authentic and beneficial contributions to the model's training. By basing rewards on the **cosine similarity** between the miners' updates and the validators' gradients, we promote alignment of miners' efforts with the overall training objectives.

### Key Points

- **Alignment of Objectives**: Miners are motivated to perform authentic training on their assigned data subsets because providing updates that closely match the true gradient direction maximizes their rewards.
- **Positive Contributions**: By submitting deltas that positively impact the model's performance on the evaluation data, miners increase their rewards.
- **Discouraging Malicious Behavior**: Contributions that deviate significantly from the true gradient (e.g., random or adversarial updates) result in lower or negative rewards.
- **Data Subset Specialization**: Miners are evaluated based on their performance on specific data subsets, encouraging them to specialize and optimize their training for those subsets.
- **Fairness**: By not revealing which model slices need to be uploaded until the end of the window, all miners are on a level playing field, preventing exploitation of the system.

### Mathematical Details

#### Notations

- **$\theta$**: Current model parameters.
- **$\delta_i$**: Delta (model update) contributed by miner **$i$**.
- **$g_i$**: Gradient of the loss with respect to the model parameters on the data subset assigned to miner **$i$**.
- **$\hat{g}_i$**: Validator's locally computed gradient on the same data subset.
- **$s_i$**: Cosine similarity score between **$\delta_i$** and **$\hat{g}_i$**.
- **$R_i$**: Reward assigned to miner **$i$**.

#### Cosine Similarity Calculation

The cosine similarity between the miner's delta and the validator's gradient is calculated as:

$$
s_i = \frac{\delta_i \cdot \hat{g}_i}{\|\delta_i\| \|\hat{g}_i\|}
$$

- **$\delta_i \cdot \hat{g}_i$**: Dot product of the miner's delta and the validator's gradient.
- **$\|\delta_i\|$** and **$\|\hat{g}_i\|$**: Euclidean norms of the miner's delta and the validator's gradient, respectively.

#### Reward Calculation

The reward for miner **$i$** is directly proportional to the cosine similarity score:

$$
R_i = \alpha \cdot s_i
$$

Where **$\alpha$** is a scaling factor determined by the network's economic parameters.

- A higher cosine similarity **$s_i$** indicates that the miner's update is closely aligned with the true gradient, resulting in a higher reward.
- If **$s_i$** is negative, it indicates that the miner's update is detrimental to the model's performance on the validation data, leading to a lower or negative reward.

#### Intuition Behind the Mechanism

- **Positive Reinforcement**: Miners are rewarded for updates that point in the same direction as the true gradient, improving the model.
- **Penalty for Divergence**: Miners submitting random or harmful updates receive lower rewards due to low or negative cosine similarity.
- **Efficient Collaboration**: This mechanism encourages miners to focus on genuine training rather than attempting to game the system.

## Installation Guide

### Prerequisites

- **Python 3.8 or higher**
- **Pip** for package management
- **Git** for cloning the repository
- **AWS Account** with access to S3 (Simple Storage Service)
- **Compute Resources**:
  - **Miner**: High-performance GPU (e.g., NVIDIA A100 or better) recommended for training.
  - **Validator**: Similar compute requirements as miners due to the need to recompute gradients.

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/unconst/bistro.git
   cd bistro
   ```

2. **Set Up AWS Credentials**

   Configure your AWS credentials to allow read and write access to your S3 bucket:

   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key_id
   export AWS_SECRET_ACCESS_KEY=your_secret_access_key
   ```

   Ensure that your S3 bucket has the necessary permissions for read and write operations.

3. **Install Dependencies**

   It's recommended to use a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Create a `.env` file or export the environment variables required by the project.

5. **Register on Bittensor Subnet**

   The system runs on a Bittensor subnet. You need to register your miner and validator.

   ```bash
   # Replace <> with your actual wallet names and hotkeys.
   btcli register --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --subtensor.network test --netuid 220
   ```

## Running the Miner and Validator

### Run the Miner

```bash
python3 miner.py \
    --wallet.name <wallet_name> \
    --wallet.hotkey <hotkey_name> \
    --subtensor.network test \
    --netuid 220 \
    --bucket <your_s3_bucket_name> \
    --device cuda
```

### Run the Validator

```bash
python3 validator.py \
    --wallet.name <wallet_name> \
    --wallet.hotkey <hotkey_name> \
    --subtensor.network test \
    --netuid 220 \
    --bucket <your_s3_bucket_name> \
    --device cuda
```

## Hardware Requirements

Given the computational intensity of training and validating neural networks, it is highly recommended to use machines equipped with high-performance GPUs like NVIDIA A100 or better. Adequate CPU resources and memory are also necessary to handle data loading and preprocessing tasks.

## Contributing

Contributions to the **BISTRO** project are welcome. Please open issues and submit pull requests for improvements and fixes.

## License

This project is licensed under the MIT License © 2024 Chakana.tech. See the [LICENSE](LICENSE) file for details.

---

**Note**: The mathematical formulations and mechanisms described are integral to ensuring the security and efficiency of the decentralized training process. By participating as a miner or validator, you contribute to a collaborative effort to advance decentralized machine learning.