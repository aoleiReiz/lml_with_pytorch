import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def synthetic_data(w_, b_, num_examples):
    """generate y = wx + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(w_)))
    y = torch.matmul(X, w_) + b_
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


class CustomDataset(Dataset):
    def __init__(self, x_, y_):
        super(CustomDataset, self).__init__()
        self.x = x_
        self.y = y_

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


class Model(nn.Module):
    def __init__(self, n_inputs_):
        super(Model, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_inputs_, 20),
            nn.Linear(20, 10),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.linear(x)


n_train, n_test, n_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((n_inputs, 1)) * 0.01, 0.05
train_X, train_y = synthetic_data(true_w, true_b, n_train)
test_X, test_y = synthetic_data(true_w, true_b, n_test)
train_dataset = CustomDataset(train_X, train_y)
test_dataset = CustomDataset(test_X, test_y)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
learning_rate = 0.001
epochs = 5
model_ = Model(n_inputs)
loss_fn_ = nn.MSELoss()
optimizer_ = torch.optim.SGD(model_.parameters(), lr=learning_rate, weight_decay=1e-3)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error:, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_data_loader, model_, loss_fn_, optimizer_)
    test_loop(test_data_loader, model_, loss_fn_)
print("Done!")


