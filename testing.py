import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

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


def torch_testing():
    state_vectors, vectors, lte = data.load_data()

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






