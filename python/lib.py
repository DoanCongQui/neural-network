"""
@ LIBRARY all algorithm neural networking
@ dq
@ Github: github.com/DoanCongQui
"""

import numpy as np
from rich.console import Console
from rich.table import Table


class active_function:
    def __init__(self) -> None:
        pass

    def _step(self, net):
        return 0 if net < 0 else 1

    def _sign(self, net):
        return -1 if net < 0 else 1

    def _linear(self, net):
        return net

    def _ReLU(self, net):
        return 0 if net < 0 else net

    def _sigmoid(self, net):
        return 1 / (1 + np.exp(-net))

    def _tanh(self, net):
        return (1 - np.exp(-2 * net)) / (1 + np.exp(-2 * net))


class NeuralNertwork:
    def __init__(self, X, W, d, teta, max_epoch):
        self.active = active_function()
        self.X = X
        self.W = W
        self.d = d
        self.teta = teta
        self.max_epoch = max_epoch
        self.console = Console()

    def display_epoch(self, epoch, epoch_data):
        """Display output"""
        table = Table(title=f"[green]Epoch {epoch}[/green]", show_lines=True)

        # Add column
        table.add_column("#", justify="center", style="bold cyan")
        table.add_column("net", justify="center", style="bold yellow")
        table.add_column("y (Output)", justify="center", style="bold green")
        table.add_column("W (Weights)", style="bold magenta")
        table.add_column("E (Error)", justify="center", style="bold red")

        # Add data
        for row in epoch_data:
            table.add_row(*row)

        self.console.print(table)

    def Perceptron(self):
        epoch = 0
        while epoch < self.max_epoch:
            epoch_data = []
            E = 0
            for i in range(self.X.shape[1]):
                net = self.W.T @ self.X[:, i]
                y = self.active._step(net)
                self.W += self.teta * (self.d[i] - y) * self.X[:, i]
                E = E + 0.5 * (self.d[i] - y) ** 2

                epoch_data.append(
                    [str(i), f"{net:.2f}", str(y), str(self.W), f"{E:.1f}"]
                )

            # Create table data
            self.display_epoch(epoch + 1, epoch_data)

            epoch += 1
            if E == 0:
                break


if __name__ == "__main__":
    X = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
    W = np.array([0.1, 0.3, 0.5])
    d = np.array([0, 0, 0, 1])

    nn = NeuralNertwork(X, W, d, 0.1, 1000)
    nn.Perceptron()
