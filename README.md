<p align="center">
]3 | _\~ ~|~ /? ()
</p>

---
# BISTRO: Bittensor Incentivized and Scalable Training with Reward Optimization
---

Welcome to the BISTRO project! This repository contains the implementation of an innovative incentive system for decentralized machine learning training using the Bittensor network.

## Overview

In BISTRO, we introduce a decentralized framework where **miners** contribute to training a shared model by processing specific subsets of data, and **validators** ensure the quality of contributions. The incentive mechanism is designed to reward miners for effectively training on their designated data subsets, promoting efficient and collaborative model improvement.

## How It Works

### Miners

- **Training**: Miners receive a designated subset of pages from the dataset for each block window. They train the model on this subset, performing a single gradient update per window.
- **Uploading Slices**: After training, miners upload a subset of their model weights (called **slices**) to an S3 bucket. The specific weights to upload are not known until the end of the window, ensuring fairness and preventing pre-uploading untrained weights.
- **Window Progression**: Miners proceed to the next window, repeating the process with a new data subset.

### Validators

- **Fetching Slices**: Validators download the slices uploaded by miners from the S3 buckets corresponding to the last window.
- **Evaluation**: Validators evaluate the performance of the miners' slices by comparing the miner's uploaded gradient to the gradient that the validator computes using the same data subset.
- **Scoring**: The reward for each miner is calculated based on the **negative difference** between the miner's gradient and the validator's recomputed gradient. This means that miners are incentivized to minimize this difference by accurately training on their assigned data.

## Mathematical Incentive Design

Let’s denote:

- \( g_m \): Gradient computed by the miner on their data subset.
- \( g_v \): Gradient computed by the validator on the same data subset.
- \( R_m \): Reward for the miner.

The reward is calculated as:

\[ R_m = - \| g_m - g_v \| \]

Miners aim to maximize their rewards by minimizing the difference between their computed gradients and the ones expected by validators. This directly incentivizes miners to genuinely train on their assigned data subsets every window, as deviating from the expected gradient reduces their reward.

## Installation Guide

### Prerequisites

- **Python 3.8 or higher**
- **Pip** for package management
- **Git** for cloning the repository
- **AWS Account** with access to S3 (Simple Storage Service)
- **Compute Resources**:
  - **Miner**: High-performance GPU (e.g., NVIDIA H100) recommended for training.
  - **Validator**: Similar compute requirements as miners due to the need to recompute gradients.

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/bistro.git
   cd bistro
   ```

2. **Set Up AWS Credentials**

   Create an S3 bucket and configure your AWS credentials:

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
   btcli register --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --subtensor.network test --netuid 212
   ```

## Running the Miner and Validator

### Run the Miner

```bash
python3 miner.py \
    --wallet.name <wallet_name> \
    --wallet.hotkey <hotkey_name> \
    --subtensor.network test \
    --netuid 212 \
    --bucket <your_s3_bucket_name> \
    --device cuda
```

### Run the Validator

```bash
python3 validator.py \
    --wallet.name <wallet_name> \
    --wallet.hotkey <hotkey_name> \
    --subtensor.network test \
    --netuid 212 \
    --bucket <your_s3_bucket_name> \
    --device cuda
```

## Hardware Requirements

Given the computational intensity of training and validating neural networks, it is highly recommended to use machines equipped with high-performance GPUs like NVIDIA H100. Adequate CPU resources and memory are also necessary to handle data loading and preprocessing tasks.

## Understanding the Incentive Mechanism

The incentive mechanism in BISTRO is designed to ensure that miners are rewarded for genuine contributions to the model's training. By making the reward proportional to the negative difference between the miner’s gradient and the validator’s gradient, we ensure the following:

- **Alignment of Objectives**: Miners are motivated to perform authentic training on their assigned data subsets, as any deviation reduces their rewards.
- **Data Subset Specialization**: Since miners are evaluated based on their performance on specific data subsets, they are encouraged to specialize and optimize their training for those subsets.
- **Fairness**: By not revealing which weights need to be uploaded until the end of the window, all miners are on a level playing field, preventing any potential exploitation of the system.

This mathematical design drives miners to consistently train on their designated data, ensuring the overall model benefits from diverse and comprehensive training across different data subsets.

## Contributing

Contributions to the BISTRO project are welcome. Please open issues and submit pull requests for improvements and fixes.

## License

This project is licensed under the MIT License © 2024 Chakana.tech. See the `LICENSE` file for details.


