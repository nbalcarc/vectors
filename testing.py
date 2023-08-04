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


def torch_testing_binary():
    # load the dataset, split into input (X) and output (y) variables
    dataset = np.loadtxt('test_data.csv', delimiter=',')
    X = dataset[:,0:8]
    y = dataset[:,8]

    print(X.shape)
    print(y.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)

    model = nn.Sequential(
        nn.Linear(8, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
        ).to(device)

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


def torch_testing_regression():
    state_vectors, vectors, lte = data.load_data()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    state_vectors = torch.from_numpy(state_vectors)
    vectors = torch.from_numpy(vectors)
    lte = torch.from_numpy(lte)

    lte_10 = lte[:, 0, :] #0 for 10, 1 for 30, 2 for 50
    print(lte_10.shape)

    # training on vectors (size 1024)
    X_train = vectors[:31].reshape(-1, 1024).to(device)
    y_train = lte_10[:31].reshape(-1, 1).to(device)
    X_test = vectors[31].reshape(-1, 1024).to(device)
    y_test = lte_10[31].reshape(-1, 1).to(device)

    print(X_train.shape)
    print(y_train.shape)

    print(y_train)

    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        #nn.Sigmoid()
        ).to(device)
    print(model)

    #loss_fn = nn.BCELoss()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)

    n_epochs = 100
    losses = np.zeros(n_epochs)
    batch_size = 10
    loss = 0  #ease type errors

    best_mse = np.inf
    best_weights = None
    history = []
    
    #training loop
    for epoch in range(n_epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            # initialize batches
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()  #update weights

        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        print(f'Finished epoch {epoch}, latest loss {loss}')
        losses[epoch] = loss

    print(losses)
        


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
    X_train = vectors[:31].reshape(-1, 1024).to(device)
    y_train = lte_10[:31].reshape(-1, 1).to(device)
    X_test = vectors[31].reshape(-1, 1024).to(device)
    y_test = lte_10[31].reshape(-1, 1).to(device)

    print(X_train.shape)
    print(y_train.shape)

    print(y_train)

    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid()
        ).to(device)
    print(model)

    #loss_fn = nn.BCELoss()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)

    n_epochs = 10
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
    #with torch.no_grad():
    #    y_pred = model(X_train)
    y_pred = model(X_train)
    accuracy = (y_pred.round() == y_train).float().mean()
    print(f"Accuracy {accuracy}")








