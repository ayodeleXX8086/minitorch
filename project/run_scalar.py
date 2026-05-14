import random
import minitorch
from minitorch.datasets import Graph


# ---------------- Linear Layer ----------------
class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []

        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                w = minitorch.Scalar(2 * (random.random() - 0.5))
                self.weights[i].append(self.add_parameter(f"weight_{i}_{j}", w))

        for j in range(out_size):
            b = minitorch.Scalar(2 * (random.random() - 0.5))
            self.bias.append(self.add_parameter(f"bias_{j}", b))

    def forward(self, inputs):
        # Initialize y with the actual Scalar objects, not Python floats
        y = [bias.value for bias in self.bias]
        for i, x in enumerate(inputs):
            for j in range(len(y)):
                y[j] = y[j] + x * self.weights[i][j].value
        return y

# ---------------- Network ----------------
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.add_module("layer1", Linear(2, hidden_layers))
        self.add_module("layer2", Linear(hidden_layers, hidden_layers))
        self.add_module("layer3", Linear(hidden_layers, 1))

    def forward(self, x):
        middle = [h.relu() for h in self.layer1.forward(x)]
        end = [h.relu() for h in self.layer2.forward(middle)]
        return self.layer3.forward(end)[0].sigmoid()

# ---------------- Training ----------------
def default_log_fn(epoch, total_loss, correct, losses):
    print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Correct: {correct}/{len(losses)}")

class ScalarTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"),
             minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))

                # Binary classification loss
                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0

                loss = -prob.log()
                loss.backward()
                total_loss += loss.data

            optim.step()
            losses.append(total_loss)

            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)

def py_torch_example(data: Graph, lr=0.01, hidden=2, log_fn=None):
    import torch
    import torch.nn as nn

    class Network(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1), nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x).squeeze()

    model = Network(hidden)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, 501):
        optimizer.zero_grad()
        X = torch.tensor(data.X, dtype=torch.float32)
        y = torch.tensor(data.y, dtype=torch.float32)
        out = model(X)
        loss = -(y * out.log() + (1 - y) * (1 - out).log()).sum()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            correct = ((out > 0.5) == y.bool()).sum().item()
            if log_fn:
                log_fn(epoch, )
            else:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Correct: {correct}/{len(y)}")


# ---------------- Main ----------------
if __name__ == "__main__":
    PTS = 105
    data = minitorch.datasets["Xor"](PTS)
    EPOCH = 500

    hidden_options = [2, 3, 5, 7, 9]
    rate_options = [0.5, 0.1, 0.05, 0.01, 0.001]

    minitorch_results = []
    pytorch_results = []

    for hidden in hidden_options:
        for rate in rate_options:
            # --- Minitorch ---
            final_loss = None
            final_correct = None

            def silent_log(epoch, total_loss, correct, losses):
                global final_loss, final_correct
                final_loss = total_loss
                final_correct = correct

            ScalarTrain(hidden).train(data, rate, max_epochs=EPOCH, log_fn=silent_log)
            minitorch_results.append({
                "hidden": hidden, "rate": rate,
                "final_loss": round(final_loss, 4),
                "correct": final_correct,
            })

            # --- PyTorch ---
            import torch
            import torch.nn as nn

            class TorchNetwork(nn.Module):
                def __init__(self, h):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(2, h), nn.ReLU(),
                        nn.Linear(h, h), nn.ReLU(),
                        nn.Linear(h, 1), nn.Sigmoid()
                    )
                def forward(self, x):
                    return self.net(x).squeeze()

            model = TorchNetwork(hidden)
            optimizer = torch.optim.SGD(model.parameters(), lr=rate)
            X = torch.tensor(data.X, dtype=torch.float32)
            y = torch.tensor(data.y, dtype=torch.float32)

            for epoch in range(1, EPOCH+1):
                optimizer.zero_grad()
                out = model(X)
                loss = -(y * out.log() + (1 - y) * (1 - out).log()).sum()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                out = model(X).clamp(1e-7, 1 - 1e-7)
                pt_correct = ((out > 0.5) == y.bool()).sum().item()
                pt_loss = round(loss.item(), 4)

            pytorch_results.append({
                "hidden": hidden, "rate": rate,
                "final_loss": pt_loss,
                "correct": int(pt_correct),
            })

    # --- Print comparison table ---
    print(f"\n===== GRID SEARCH COMPARISON ({EPOCH} epochs) =====")
    print(f"{'HIDDEN':<8} {'RATE':<8} {'MT_LOSS':<12} {'MT_CORRECT':<14} {'PT_LOSS':<12} {'PT_CORRECT':<10}")
    print("-" * 65)
    for mt, pt in zip(minitorch_results, pytorch_results):
        print(
            f"{mt['hidden']:<8} {mt['rate']:<8} "
            f"{mt['final_loss']:<12} {mt['correct']}/{PTS:<11} "
            f"{pt['final_loss']:<12} {pt['correct']}/{PTS}"
        )