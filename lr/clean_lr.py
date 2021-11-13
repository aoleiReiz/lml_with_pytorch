import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def synthetic_data(w, b, num_examples_):
    X = torch.normal(0, 1, (num_examples_, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_reg = nn.Sequential(
            nn.Linear(2, 1)
        )

    def forward(self, x):
        logits = self.linear_reg(x)
        return logits


class CustomLrDataset(Dataset):

    def __init__(self, features_, labels_):
        self.features = features_
        self.labels = labels_

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]


# data generation
num_examples = 1000
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, num_examples)

train_data = CustomLrDataset(features[:800], labels[:800])
test_data = CustomLrDataset(features[800:], labels[800:])


learning_rate = 0.03
batch_size = 10
epochs = 3
model = Net()

train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=10, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


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


def test_loop(dataloader, model_, loss_fn_):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model_(X)
            test_loss += loss_fn_(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error:, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")