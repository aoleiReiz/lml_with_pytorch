import random

import torch
import matplotlib.pyplot as plt


def synthetic_data(w, b, num_examples):
    """generate y = wx + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size_, features_, labels_):
    num_examples = len(features_)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size_):
        batch_indices = torch.tensor(indices[i: min(i + batch_size_, num_examples)])
        yield features_[batch_indices], labels_[batch_indices]


# data generation
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# batch
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, "\n", y)
    break

# initialization
w = torch.normal(0, 0.0, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# model
def linear_reg(x_, w_, b_):
    """linear regression"""
    return torch.matmul(x_, w_,) + b_


# loss
def squared_loss(y_hat_, y_):
    """squared_loss"""
    return (y_hat_ - y_.reshape(y_hat_.shape)) ** 2 / 2


# optimization
def sgd(params_, lr_, batch_size_):
    """gradient descent"""
    with torch.no_grad():
        for param in params_:
            param -= lr_ * param.grad / batch_size_
            param.grad.zero_()


# training
lr = 0.03
num_epochs = 3
net = linear_reg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f"epoch {epoch + 1}, loss {float(train_l.mean())}")


predictions = linear_reg(features, w, b)

plt.scatter(features[:, (1,)].detach().numpy(), labels.detach().numpy(), marker=".", color="green")
plt.scatter(features[:, (1,)].detach().numpy(), predictions.detach().numpy(), marker="_", color="red")
plt.show()
