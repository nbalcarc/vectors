import pickle
import numpy as np
import numpy.typing as npt


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


