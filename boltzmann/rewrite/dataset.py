import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from boltzmann.rewrite.model import MODEL_TYPE
from boltzmann.rewrite.settings import general_settings, tiny_nn_settings


class DatasetFactory:
    @staticmethod
    def create_dataset(model_type: MODEL_TYPE) -> tuple[Dataset, Dataset]:
        match model_type:
            case "cifar10_cnn":
                # Define data transformation (e.g., normalization)
                transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                        ),
                    ]
                )

                train_dataset = datasets.CIFAR10(
                    root="./data", train=True, download=True, transform=transform
                )
                val_dataset = datasets.CIFAR10(
                    root="./data", train=False, download=True, transform=transform
                )
                return train_dataset, val_dataset
            case "tiny_nn" | "two_neuron_network":
                dataset = LinearRegressionDataset(
                    general_settings.data_size, tiny_nn_settings.input_size
                )
                # Split dataset into training and validation sets
                train_size = int(
                    (1 - general_settings.val_data_fraction) * len(dataset)
                )
                val_size = len(dataset) - train_size
                return torch.utils.data.random_split(dataset, [train_size, val_size])
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")


class LinearRegressionDataset(Dataset):
    def __init__(self, num_samples: int, num_features: int):
        # Generate features and targets
        self.features, self.targets = self._generate_linear_data(
            num_samples, num_features
        )

    def _generate_linear_data(
        self, num_samples: int, num_features: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Generate random features
        X = torch.rand(num_samples, num_features)

        # Generate random coefficient(s) for linear relationship
        coefficients = (torch.rand(num_features) - 0.5) * 10
        coefficients = 10 * torch.ones_like(coefficients)

        # Normalize features
        X = (X - X.mean(dim=0)) / X.std(dim=0)

        # Generate targets
        y = X @ coefficients + torch.randn(num_samples) * 1
        return X, y.unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
