import matplotlib.pyplot as plt
import torch

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

    print(lte.shape)
    print(type(lte))
    print(lte.type())
    print(lte.dtype)



