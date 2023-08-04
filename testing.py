import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import data


def matplotlib_testing():
    x = [10, 20, 30, 40]
    y = [20, 30, 40, 50]

    plt.plot(x, y)
    plt.axis([0, 50, 0, 60])  #min x, max x, min y, max y
    plt.ylabel("y-axis")
    plt.xlabel("x-axis")
    plt.title("Simple Post1")
    plt.savefig("output.png")


def matplotlib_testing1():
    state_vectors, vectors, lte = data.load_data()

    zeroes = [i for i in range(1, lte[0][0].shape[0]+1)]
    plt.plot(zeroes, lte[0, 0, :], label = "original")
    altered = lte + 1
    plt.plot(zeroes, altered[0, 0, :], label = "plus one")
    plt.title("aaaaaaaa")
    plt.legend(loc = "best")
    plt.savefig("output.png")


def torch_testing_basic():
    # load the dataset, split into input (X) and output (y) variables
    dataset = np.loadtxt('test_data.csv', delimiter=',')
    X = dataset[:,0:8]
    y = dataset[:,8]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# define the model
    class PimaClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden1 = nn.Linear(8, 12)
            self.act1 = nn.ReLU()
            self.hidden2 = nn.Linear(12, 8)
            self.act2 = nn.ReLU()
            self.output = nn.Linear(8, 1)
            self.act_output = nn.Sigmoid()

        def forward(self, x):
            x = self.act1(self.hidden1(x))
            x = self.act2(self.hidden2(x))
            x = self.act_output(self.output(x))
            return x

    model = PimaClassifier()
    print(model)

# train the model
    loss_fn   = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100
    batch_size = 10

    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# compute accuracy
    y_pred = model(X)
    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accuracy {accuracy}")

# make class predictions with the model
    predictions = (model(X) > 0.5).int()
    for i in range(5):
        print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))



def torch_testing():
    state_vectors, vectors, lte = data.load_data()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    state_vectors = torch.from_numpy(state_vectors)
    vectors = torch.from_numpy(vectors)
    lte = torch.from_numpy(lte)

    lte_10 = lte[:, 0, :] #0 for 10, 1 for 30, 2 for 50
    print(lte_10.shape)

    # training on vectors (size 1024)
    X_train = vectors[:31].reshape(-1, 1024)
    y_train = lte_10[:31].reshape(-1, 1)
    X_test = vectors[31].reshape(-1, 1024)
    y_test = lte_10[31].reshape(-1, 1)

    print(X_train.shape)
    print(y_train.shape)

    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.LeakyReLU(),
        nn.Linear(2048, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid()
        )
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    n_epochs = 100
    batch_size = 10
    loss = 0  #ease type warnings

    for epoch in range(n_epochs):
        for i in range(0, len(X_train), batch_size):
            Xbatch = X_train[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y_train[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_pred = model(X_train)

    accuracy = (y_pred.round() == y_train).float().mean()
    print(f"Accuracy {accuracy}")








