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

## Understanding the Incentive Mechanism

The incentive mechanism in BISTRO is designed to ensure that miners are rewarded for genuine contributions to the model's training. By making the reward proportional to the **negative estimated change in loss** when a miner's contribution is removed, we ensure the following:

- **Alignment of Objectives**: Miners are motivated to perform authentic training on their assigned data subsets, as contributing beneficial updates that improve the model's performance maximizes their rewards.

- **Provide Beneficial Updates**: By submitting slices that positively impact the model's performance on the evaluation data, miners increase their rewards.

- **Avoid Harmful Updates**: Since contributions that negatively impact the model (i.e., increasing the loss) result in lower rewards, miners are discouraged from submitting detrimental updates.

- **Data Subset Specialization**: Miners are evaluated based on their performance on specific data subsets, encouraging them to specialize and optimize their training for those subsets.

- **Fairness**: By not revealing which weights need to be uploaded until the end of the window, all miners are on a level playing field, preventing any potential exploitation of the system.

This mathematical design drives miners to consistently train on their designated data subsets, ensuring the overall model benefits from diverse and comprehensive training across different portions of the dataset.

### Mathematical Incentive Design

Let's denote:

- $\theta$: The current model parameters.
- $s_i$: The parameter slice (model update) contributed by miner $i$.
- $M$: Total number of miner slices being aggregated.
- $\delta\theta_i$: The perturbation vector representing the change to parameters if miner $i$'s slice is removed from the aggregated model.
- $g$: Gradient of the loss with respect to the model parameters on the evaluation data.
- $H$: Approximation of the Hessian matrix (we use the diagonal of the Fisher Information Matrix, $H \approx \text{diag}(g^2)$ ).
- $\Delta L_i$: Estimated change in loss if miner $i$'s slice is removed.
- $R_i$: Reward assigned to miner $i$.

#### Perturbation Vector Calculation

The perturbation vector $\delta\theta_i$ is calculated as:

$$
\delta\theta_i = \frac{s_i}{M - 1} - \theta
$$

Where:

- $\frac{s_i}{M - 1}$ adjusts the miner's slice to account for the aggregation without miner $i$.
- $\theta$ is the average model parameters including all slices.

#### Loss Change Estimation

The estimated change in loss $\Delta L_i$ when miner $i$'s slice is removed is computed using a second-order Taylor series approximation:

$$
\Delta L_i \approx g^\top \delta\theta_i + \frac{1}{2} \delta\theta_i^\top H \delta\theta_i
$$

- **First-Order Term ($g^\top \delta\theta_i$)**: Represents the linear impact of the perturbation on the loss.
- **Second-Order Term ($\frac{1}{2} \delta\theta_i^\top H \delta\theta_i$)**: Accounts for the curvature of the loss surface.

##### Reward Calculation

The reward for miner $i$ is then determined based on the negative of the estimated loss change:

$$
R_i = -\Delta L_i
$$

- Miners aim to **maximize** their rewards by **minimizing** $\Delta L_i$, which corresponds to contributing slices that **improve** the model (i.e., reduce the loss on the evaluation data).


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
   git clone https://github.com/unconst/bistro.git
   cd bistro
   ```

2. **Set Up AWS Credentials**

   Create an S3 bucket and configure your AWS credentials:

   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key_id
   export AWS_SECRET_ACCESS_KEY=your_secret_access_key
   ```

   Ensure that your S3 bucket has the necessary permissions for read and write operations. It is important 

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

## Contributing

Contributions to the BISTRO project are welcome. Please open issues and submit pull requests for improvements and fixes.

## License

This project is licensed under the MIT License © 2024 Chakana.tech. See the `LICENSE` file for details.
