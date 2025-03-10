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
    def __init__(self, X, W, V, d, teta, max_epoch):
        self.active = active_function()
        self.X = X
        self.W = W
        self.V = V.T
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
        table.add_column("W (Weights Out)", style="bold magenta")
        table.add_column("E (Error)", justify="center", style="bold red")
        table.add_column("V (Weights Hidden)", style="bold magenta")

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
                    [str(i+1), f"{net:.3f}", str(y), str(self.W), f"{E:.1f}"]
                )

            # Create table data
            self.display_epoch(epoch + 1, epoch_data)

            epoch += 1
            if E == 0:
                break
    
    def Adaline(self, epsilon):
        epoch = 0
        while epoch < self.max_epoch:
            epoch_data = []
            E = 0
            for i in range(self.X.shape[1]):
                net = self.W.T @ self.X[:, i]
                y = net
                self.W += self.teta * (self.d[i] - y) * self.X[:, i]
                E = E + 0.5 * (self.d[i] - y) ** 2

                epoch_data.append(
                    [str(i+1), f"{net:.3f}", str(y), str(self.W), f"{E:.1f}"]
                )

            # Create table data
            self.display_epoch(epoch + 1, epoch_data)

            epoch += 1
            if E < epsilon:
                break
    
    def Backpropagation(self, epsilon):

        # ori_display = self.display_epoch

        # def display(epoch, epoch_data):
        #     ori_display(epoch, epoch_data)
        #     table = Table(title=f"[green]Epoch {epoch}[/green]", show_lines=True)
        #     table.add_column("V (Weights)", style="bold magenta")

        # self.display_epoch = ori_display

        epoch = 0
        while epoch < self.max_epoch:
            epoch_data = []
            E = 0
            for i in range(self.X.shape[1]):
                # Step 2: 
                net_h = self.V.T @ self.X[:, i]
                # y_h = self.active._tanh(net_h)
                y_h = self.active._sigmoid(net_h)
                y_h = np.append(1, y_h)

                net_o = self.W.T @ y_h
                y_o = net_o
                # ---------------------------------

                # Step 3
                E += 1/2 * (self.d[i] - y_o)**2 
                # ---------------------------------

                # Step 4
                delta_o = self.d[i] - y_o

                delta_o = np.round(delta_o, 3)
                y_h = np.round(y_h, 3)

                self.W += self.teta * delta_o * y_h
                # ---------------------------------

                # Step 5
                W = self.W[1:]
                y = y_h[1:]
                # delta_h = (delta_o * W) * (1 - y**2)
                delta_h = (delta_o * W) * y*(1 - y)
                delta_h = np.round(delta_h, 3)

                for j in range(self.V.shape[1]):
                    # self.v[:, j] += eta * delta_h[j] * x.T
                    self.V[:, j] += (self.teta * delta_h[j] * self.X[:, i])
                # ---------------------------------
                
                self.W = np.round(self.W, 3)
                self.V = np.round(self.V, 3)

                epoch_data.append(
                    [str(i+1), str(net_h), f"{y_o:.3f}", str(self.W), f"{E:.3f}", str(self.V)]
                )

            # Create table data
            self.display_epoch(epoch + 1, epoch_data)

            epoch += 1
            if E < epsilon:
                break


if __name__ == "__main__":
    X = np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]])
    W = np.array([0.1, 0.3, 0.5])
    d = np.array([0, 0, 0, 1])

    nn = NeuralNertwork(X, W, d, 0.1, 1000)
    nn.Perceptron()
