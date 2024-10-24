import json
from copy import deepcopy
from datetime import datetime
from typing import Callable, Literal

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

import wandb
from boltzmann.rewrite.logger import general_logger, metrics_logger
from boltzmann.rewrite.settings import device, tiny_nn_settings


class MinerSlice(BaseModel):
    miner_id: int = Field(..., description="Unique identifier for the miner")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of when the slice was uploaded",
    )
    data: torch.Tensor = Field(..., description="Slice of the model's parameters")

    class Config:
        arbitrary_types_allowed = True


MODEL_TYPE = Literal[
    "two_neuron_network", "tiny_nn", "cifar10_cnn", "resnet18", "densenet", "deit-b"
]


class ModelFactory:
    @staticmethod
    def create_model(model_type: MODEL_TYPE):
        loss_transformation = None
        match model_type:
            case "deit-b":
                # Load the pretrained DeiT model
                torch_model = timm.create_model(
                    "deit_base_patch16_224", pretrained=False, num_classes=10
                )

                # Define loss and optimizer
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-4)
            case "densenet":
                torch_model = models.densenet121(pretrained=False)
                torch_model.classifier = torch.nn.Linear(
                    torch_model.classifier.in_features, 10
                )
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(
                    torch_model.parameters(), lr=0.001, weight_decay=1e-4
                )
            case "resnet18":
                torch_model = models.resnet18()
                torch_model.fc = torch.nn.Linear(torch_model.fc.in_features, 10)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(
                    torch_model.parameters(), lr=0.001, weight_decay=1e-4
                )
            case "cifar10_cnn":
                torch_model = CIFAR10CNN()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(torch_model.parameters(), lr=0.001)
            case "two_neuron_network":
                torch_model = OneNeuronNetwork()
                criterion = nn.MSELoss()
                optimizer = optim.SGD(torch_model.parameters(), lr=0.01)
                return Model(
                    torch_model,
                    optimizer,
                    criterion,
                )
            case "tiny_nn":
                torch_model = TinyNeuralNetwork(
                    tiny_nn_settings.input_size,
                    tiny_nn_settings.hidden_size,
                    tiny_nn_settings.output_size,
                )
                optimizer = optim.AdamW(torch_model.parameters(), lr=0.01)
                criterion = nn.MSELoss()  # Mean squared error for regression

                def loss_transformation(
                    loss: torch.Tensor, torch_model: nn.Module
                ) -> torch.Tensor:
                    # Add L1 regularization to the loss
                    # Regularization factor (L1 penalty strength)
                    l1_lambda = 0
                    l1_norm = sum(p.abs().sum() for p in torch_model.parameters())
                    return loss + l1_lambda * l1_norm  # Add L1 penalty

            case _:
                raise ValueError(f"Unsupported model type: {model_type}")
        return Model(
            torch_model,
            optimizer,
            criterion,
            loss_transformation=loss_transformation,
        )


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # CIFAR10 has 10 classes
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TinyNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TinyNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.0)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        return self.fc2(x)


class OneNeuronNetwork(nn.Module):
    def __init__(self):
        super(OneNeuronNetwork, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input and one output

    def forward(self, x):
        return self.linear(x)


class Model:
    def __init__(
        self,
        torch_model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.modules.loss._Loss,
        loss_transformation: Callable | None = None,
    ):
        self.torch_model = torch_model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_transformation = loss_transformation
        self.val_losses = torch.tensor([])
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        self.log_data = {}

    def add_metric_to_log(self, metric_name: str, value: float) -> None:
        self.log_data[metric_name] = value
        wandb.log(self.log_data, commit=False)

    def commit_log(self) -> None:
        wandb.log(self.log_data, commit=True)

    def val_step(self, data: torch.Tensor) -> tuple[float, torch.types.Number, int]:
        general_logger.debug("Validation step")
        inputs, targets = data

        # Compute loss without updating gradients
        with torch.no_grad():
            outputs = self.torch_model(inputs)
            loss = self.criterion(outputs, targets)

        # Calculate accuracy (assuming classification)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max logit
        correct = (predicted == targets).sum().item()  # Count correct predictions
        total = targets.size(0)  # Number of samples in the batch

        return loss.item(), correct, total  # Return loss, correct preds, and total

    def validate(self, val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]):
        """
        Run validation over the entire validation dataset.
        This method aggregates loss and accuracy over all validation batches.
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        self.val_losses = torch.tensor([])  # Reset validation losses

        # Loop through all validation batches
        for features, targets in val_loader:
            val_batch = (features.to(device), targets.to(device))
            loss, correct, total = self.val_step(val_batch)
            total_loss += loss * total  # Accumulate weighted loss
            total_correct += correct  # Accumulate correct predictions
            total_samples += total  # Accumulate total number of samples

        # Calculate average loss and accuracy over the entire validation set
        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        # Log metrics
        self.add_metric_to_log("val_loss", avg_loss)
        self.add_metric_to_log("val_acc", avg_accuracy)
        self.commit_log()
        return avg_loss, avg_accuracy

    def log_metrics_to_file(self, metric_name: str, value: float) -> None:
        """Log and store metrics in JSON format."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

        # Log the entire metrics dictionary as a JSON string
        metrics_logger.info(json.dumps({metric_name: value}))

    def train_step(self, data: tuple[torch.Tensor, torch.Tensor]):
        general_logger.debug("Training one step")
        inputs, targets = data

        # Zero gradients from previous step
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.torch_model(inputs)

        # Compute loss
        loss = self.criterion(outputs, targets)
        if self.loss_transformation:
            loss = self.loss_transformation(loss, self.torch_model)

        # Backward pass (compute gradients)
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss.item()

    def update_with_slice(
        self, slice: torch.Tensor, slice_indices: torch.Tensor
    ) -> None:
        """Insert slice into the model."""
        # Flatten the own model
        own_model_params = torch.nn.utils.parameters_to_vector(
            self.torch_model.parameters()
        )
        own_model_params[slice_indices] = slice
        torch.nn.utils.vector_to_parameters(  # Needed? Or does the previous line suffice?
            own_model_params, self.torch_model.parameters()
        )

    def aggregate_slices(
        self, slices: list[torch.Tensor], slice_indices: torch.Tensor
    ) -> nn.Module:
        # Flatten parameters
        params = torch.nn.utils.parameters_to_vector(self.torch_model.parameters())

        # Aggregate
        slices_elementwise_avg = torch.mean(torch.stack(slices), dim=0)

        # Update model
        params[slice_indices] = slices_elementwise_avg
        new_torch_model = deepcopy(self.torch_model)
        torch.nn.utils.vector_to_parameters(params, new_torch_model.parameters())
        return new_torch_model

    def get_params(self):
        # Get a dictionary of model parameters
        return {
            name: param.clone() for name, param in self.torch_model.named_parameters()
        }

    def update_params(self, param_slice):
        # Update model parameters using received slice
        with torch.no_grad():
            for name, param in self.torch_model.named_parameters():
                if name in param_slice:
                    param.copy_(param_slice[name])

    def compute_loss(self, data):
        # Compute loss for validation
        inputs, targets = data

        with torch.no_grad():
            outputs = self.torch_model(inputs)
            loss = self.criterion(outputs, targets)

        return loss.item()

    def sum_params(self):
        return torch.tensor([p.sum() for p in self.torch_model.parameters()]).sum()

    def num_params(self):
        return sum(p.numel() for p in self.torch_model.parameters())
