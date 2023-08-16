import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import data

def predict_data():
    state_vectors, vectors, lte = data.load_data()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    state_vectors = torch.from_numpy(state_vectors)
    vectors = torch.from_numpy(vectors)
    lte = torch.from_numpy(lte)

    lte_10 = lte[:, 0, :] #0 for lte10, 1 for lte30, 2 for lte50
    lte_30 = lte[:, 1, :]
    lte_50 = lte[:, 2, :]
    #print(lte_10.shape)

    for i_setting in range(6):
        match i_setting:
            case 0:
                cur_lte = lte_10
                cur_vec = state_vectors
            case 1:
                cur_lte = lte_30
                cur_vec = state_vectors
            case 2:
                cur_lte = lte_50
                cur_vec = state_vectors
            case 3:
                cur_lte = lte_10
                cur_vec = vectors
            case 4:
                cur_lte = lte_30
                cur_vec = vectors
            case _:
                cur_lte = lte_50
                cur_vec = vectors

        if i_setting < 3:
            # training on state_vectors (size 2048)
            X_train = cur_vec[:31].reshape(-1, 2048).to(device)
            y_train = cur_lte[:31].reshape(-1, 1).to(device)
            X_test = cur_vec[31].reshape(-1, 2048).to(device)
            y_test = cur_lte[31].reshape(-1, 1).to(device)

            model = nn.Sequential(
                    nn.Linear(2048, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1),
                    ).to(device)
        else:
            # training on vectors (size 1024)
            X_train = cur_vec[:31].reshape(-1, 1024).to(device)
            y_train = cur_lte[:31].reshape(-1, 1).to(device)
            X_test = cur_vec[31].reshape(-1, 1024).to(device)
            y_test = cur_lte[31].reshape(-1, 1).to(device)

            model = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1),
                    ).to(device)

        #print(model)
        #print(X_train.shape)
        #print(y_train.shape)
        #print(y_train)

        #loss_fn = nn.BCELoss()
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.0001)

        n_epochs = 100
        history = np.zeros(n_epochs)
        batch_size = 10
        loss = 0  #ease type errors

        print(f"=== Training model {i_setting + 1} of 6 === ")

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
        plt.clf()
        plt.plot(np.arange(n_epochs), history, label = "original")
        plt.title("Loss Over Time")

        file = ""
        if i_setting < 3:
            file += f"output{i_setting}_state_vectors_"
        else:
            file += f"output{i_setting}_vectors_"
        if i_setting % 3 == 0:
            file += "lte10.png"
        elif i_setting % 3 == 1:
            file += "lte30.png"
        else:
            file += "lte50.png"
            
        plt.savefig(file)




