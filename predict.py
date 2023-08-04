import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import data

def predict_data():
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
    history = np.zeros(n_epochs)
    batch_size = 10
    loss = 0  #ease type errors

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
        true_loss = loss_fn(y_pred, y_test)
        print(f'Finished epoch {epoch}, latest loss {true_loss}')
        history[epoch] = true_loss

    # graph results
    plt.plot(np.arange(n_epochs), history, label = "original")
    plt.title("loss over time")
    plt.savefig("output.png")




