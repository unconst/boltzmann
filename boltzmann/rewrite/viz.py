import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class InteractivePlotter:
    def __init__(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        self.xlim = xlim
        self.ylim = ylim
        (
            self.fig,
            self.ax,
            self.line_data,
            self.line_model,
        ) = self.set_up_interactive_plotting()

    def set_up_interactive_plotting(self) -> tuple[Figure, Axes, Line2D, Line2D]:
        # Enable interactive mode
        plt.ion()

        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        # Placeholder for the plot line
        (line_data,) = ax.plot([], [], "c.", label="data")
        (line_model,) = ax.plot([], [], "m", label="model")
        # Set up the axis limits and labels
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.legend()
        return fig, ax, line_data, line_model

    def plot_data_and_model(
        self,
        *,
        torch_model: torch.nn.Module,
        features: torch.Tensor,
        targets: torch.Tensor,
        num_points: int = 1000,
    ) -> None:
        x_for_model = torch.linspace(self.xlim[0], self.xlim[1], num_points).unsqueeze(
            1
        )
        self.line_data.set_xdata(features)
        self.line_model.set_xdata(x_for_model)
        with torch.no_grad():
            output = torch_model(x_for_model)
        self.line_data.set_ydata(targets)
        self.line_model.set_ydata(output.detach())

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def plot_scores(scores: dict[int, torch.Tensor]) -> None:
    for miner_id, values in scores.items():
        rounds = range(1, len(values) + 1)  # X-axis (communication rounds)
        plt.plot(
            rounds, values.numpy(), label=f"Miner {miner_id}"
        )  # Plot each miner's scores

    plt.title("Scores per Communication Round for Each Miner")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Scores")
    plt.legend(title="Miner ID")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
