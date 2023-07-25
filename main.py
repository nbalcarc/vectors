import pickle
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def read_data() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Reads and returns data from the pickle files."""

    with open("state_vectors.pkl", 'rb') as file:
        raw_state_vectors = pickle.load(file)["Riesling"]["trial_0"]  #double nested dictionary
    state_vectors = np.zeros((32, 252, 2048), dtype = np.float32)
    for i in range(32):
        state_vectors[i] = raw_state_vectors[str(i)]  #numpy seems to automatically remove redundant 1D column, [0, :, :]

    with open("vectors.pkl", 'rb') as file:
        raw_vectors = pickle.load(file)["Riesling"]["trial_0"]
    vectors = np.zeros((32, 252, 1024), dtype = np.float32)
    for i in range(32):
        vectors[i] = raw_vectors[str(i)]

    with open("LTE.pkl", 'rb') as file:
        raw_lte: dict = pickle.load(file)["Riesling"]["trial_0"]
    lte = np.zeros((32, 3, 252), dtype = np.float32)
    for i in range(32):
        col = raw_lte[str(i)]
        lte[i][0] = col['10']
        lte[i][1] = col['50']
        lte[i][2] = col['90']

    return state_vectors, vectors, lte


def save_data():
    """Saves the numpy arrays."""
    state_vectors, vectors, lte = read_data()

    np.save("state_vectors.npy", state_vectors)
    np.save("vectors.npy", vectors)
    np.save("lte.npy", lte)


def load_data() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Loads the numpy arrays."""
    state_vectors: npt.NDArray[np.float32] = np.load("state_vectors.npy")
    vectors: npt.NDArray[np.float32] = np.load("vectors.npy")
    lte: npt.NDArray[np.float32] = np.load("lte.npy")

    return state_vectors, vectors, lte


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
    state_vectors, vectors, lte = load_data()

    zeroes = [i for i in range(1, lte[0][0].shape[0]+1)]
    plt.plot(zeroes, lte[0, 0, :], label = "original")
    altered = lte + 1
    plt.plot(zeroes, altered[0, 0, :], label = "plus one")
    plt.title("aaaaaaaa")
    plt.legend(loc = "best")
    plt.savefig("output.png")


def main():
    """Main entry point."""
    state_vectors, vectors, lte = load_data()

    print(f"state_vectors: {state_vectors.shape}")
    print(f"vectors: {vectors.shape}")
    print(f"lte: {lte.shape}")

    matplotlib_testing1()


if __name__ == "__main__":
    main()

