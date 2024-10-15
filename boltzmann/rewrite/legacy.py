import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer

import wandb
from boltzmann.rewrite.settings import general_settings

app = typer.Typer()


def extract_metrics_from_log(log_file: Path) -> dict[str, torch.Tensor]:
    """Extract all metrics (val_loss, val_acc, etc.) from a metrics.log file."""
    metrics = {}

    with open(log_file, "r") as f:
        for line in f:
            # Parse each line as JSON
            try:
                log_entry = json.loads(
                    line.split(" - ")[-1]
                )  # Parse JSON part of the line

                for metric, value in log_entry.items():
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric].append(value)
            except json.JSONDecodeError:
                continue  # Skip non-metric lines

    # Convert lists to tensors
    return {metric: torch.tensor(values) for metric, values in metrics.items()}


def gather_metrics(log_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Recursively gather all metrics from all metrics.log files in log_dir."""
    metrics_hyperparam = {}

    # Traverse all subdirectories
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file == "metrics.log":
                # Extract the hyperparam value from the parent directory (e.g., True:1000)
                hyperparam_value = Path(root).name
                log_file = Path(root) / file

                # Extract all metrics from the metrics log file
                metrics = extract_metrics_from_log(log_file)

                # Store the metrics under the corresponding hyperparam_value
                metrics_hyperparam[hyperparam_value] = metrics
    return metrics_hyperparam


def plot_and_save_metrics(
    metrics_hyperparam: dict[str, dict[str, torch.Tensor]],
    output_dir: Path,
    hyperparam_name: str = "Compression Factor",
) -> None:
    """Plot all metrics and save each plot to a file."""
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get a list of all available metrics
    available_metrics = set()
    for metrics in metrics_hyperparam.values():
        available_metrics.update(metrics.keys())

    # Plot each metric separately
    for metric in available_metrics:
        plt.figure()  # Create a new figure for each metric
        for hyperparam_value, metrics in metrics_hyperparam.items():
            if metric in metrics:
                plt.plot(
                    metrics[metric],
                    label=hyperparam_value,
                    linestyle="dashed" if "False" in hyperparam_value else "solid",
                    linewidth=1,  # Set the line thickness (default is 1.5)
                )

        # Customize and save the plot
        plt.title(f"{metric} for the validator model")
        plt.xlabel("Communication Rounds")
        plt.ylabel(metric)
        plt.legend(title=hyperparam_name)
        plt.tight_layout()

        # Save the plot to a file
        plot_file = output_dir / f"{metric}.png"
        plt.savefig(plot_file, dpi=600)
        plt.close()  # Close the figure to free memory


def get_latest_timestamp(logs_path: Path) -> Path:
    """Find the latest timestamped folder in the logs directory."""
    subdirs = [d for d in logs_path.iterdir() if d.is_dir()]
    latest_dir = max(
        subdirs, key=lambda d: datetime.strptime(d.name, "%Y-%m-%d %H:%M:%S.%f")
    )
    return latest_dir


@app.command()
def plot_and_save_metrics_cli(timestamp: str | None = None):
    logs_path = general_settings.results_dir / "logs"

    # If no timestamp is provided, use the latest available
    if timestamp:
        log_dir = logs_path / timestamp
    else:
        log_dir = get_latest_timestamp(logs_path)

    output_dir = general_settings.results_dir / "plots"

    # Gather all metrics from the logs
    metrics_hyperparam = gather_metrics(log_dir)

    # Plot and save all metrics to the output directory
    plot_and_save_metrics(metrics_hyperparam, output_dir)


@app.command()
def log_metrics_to_wandb(
    group: str | None = None,
):
    """Log the metrics from the logs directory to WandB."""

    # Path to the directory containing the logs
    log_dir = Path(f"results/{group}/logs")

    # Loop through all log directories (True/False with compression factors)
    for subdir in log_dir.iterdir():
        if subdir.is_dir():
            compression_factor = subdir.name  # Extract directory name (e.g., True:1000)
            same_model_init, compression = compression_factor.split(":")
            log_file = subdir / "metrics.log"

            if log_file.exists():
                # Create a new run for each hyperparameter combination but group them
                run_name = f"{'Same Init' if same_model_init == 'True' else 'Diff Init'}: Compression {compression}"
                wandb.init(project="chakana", name=run_name, group=group)

                # Extract metrics using the existing function
                metrics = extract_metrics_from_log(log_file)

                # Convert the tensors to Python types and log the metrics
                for comround in range(len(next(iter(metrics.values())))):
                    # Log compression factor and independent initialization info
                    log_data: dict[str, str | torch.types.Number] = {
                        "compression_factor": compression,
                        "independent_initialization": same_model_init,
                    }

                    # Log each metric
                    for metric_name, values in metrics.items():
                        log_data[metric_name] = values[comround].item()

                    # Log this data point to WandB
                    wandb.log(log_data, step=comround)

                # Finish the WandB run for this hyperparameter setting
                wandb.finish()


if __name__ == "__main__":
    app()
