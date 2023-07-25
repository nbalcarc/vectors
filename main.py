import pickle
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import bz2
import sys


def read_data() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Reads and returns data from the pickle files."""

    with open("state_vectors.pkl", 'rb') as file:
        raw_state_vectors = pickle.load(file)["Riesling"]["trial_0"]  #double nested dictionary
    state_vectors = np.zeros((32, 252, 2048), dtype = np.float32)
    for i in range(32):
        state_vectors[i] = raw_state_vectors[str(i)]  #numpy seems to automatically remove redundant 1D column, [0, :, :]
        #print(sys.getsizeof(raw_state_vectors[str(i)]))
        #print(sys.getsizeof(state_vectors[i]))

    print(sys.getsizeof(state_vectors))
    print(state_vectors.size)
    print(sys.getsizeof(state_vectors[1]))
    print(state_vectors[1].size)
    #print(type(raw_state_vectors["1"]))
    #print(raw_state_vectors["1"].dtype)
    #print(raw_state_vectors["1"].shape)
    #print(sys.getsizeof(raw_state_vectors["1"]))
    #print(sys.getsizeof(raw_state_vectors["1"]))
    #print(sys.getsizeof(state_vectors[1]))
    #print(sys.getsizeof(raw_state_vectors) + sys.getsizeof(raw_state_vectors['1']))
    #print(sys.getsizeof(state_vectors))

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


#def pickle_data():
#    """Pickles the input data as efficient numpy arrays."""
#    state_vectors, vectors, lte = read_data()
#
#    with open("state_vectors_np.pkl", "wb") as file:
#        pickle.dump(state_vectors, file)
#
#    with open("vectors_np.pkl", "wb") as file:
#        pickle.dump(vectors, file)
#
#    with open("lte_np.pkl", "wb") as file:
#        pickle.dump(lte, file)


#def pickle_data_compress():
#    """Pickles and compresses the input data as efficient numpy arrays."""
#    state_vectors, vectors, lte = read_data()
#
#    with bz2.BZ2File("cnp_state_vectors.pkl", "w") as file:
#    #with open("np_state_vectors.pkl", "wb") as file:
#        pickle.dump(state_vectors, file)
#
#    with bz2.BZ2File("cnp_vectors.pkl", "w") as file:
#    #with open("np_vectors.pkl", "wb") as file:
#        pickle.dump(vectors, file)
#
#    with bz2.BZ2File("cnp_lte.pkl", "w") as file:
#    #with open("np_lte.pkl", "wb") as file:
#        pickle.dump(lte, file)


#def unpickle_data() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
#    """Retrieves data from pickled numpy arrays."""
#    with open("state_vectors_np.pkl", "rb") as file:
#        state_vectors: npt.NDArray[np.float32] = pickle.load(file)
#
#    with open("vectors_np.pkl", "rb") as file:
#        vectors: npt.NDArray[np.float32] = pickle.load(file)
#
#    with open("lte_np.pkl", "rb") as file:
#        lte: npt.NDArray[np.float32] = pickle.load(file)
#    
#    return state_vectors, vectors, lte


#def unpickle_data_uncompress() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
#    """Retrieves compressed data from pickled numpy arrays."""
#    with bz2.BZ2File("cnp_state_vectors.pkl", "r") as file:
#    #with open("np_state_vectors.pkl", "rb") as file:
#        state_vectors: npt.NDArray[np.float64] = pickle.load(file)
#
#    with bz2.BZ2File("cnp_vectors.pkl", "r") as file:
#    #with open("np_vectors.pkl", "rb") as file:
#        vectors: npt.NDArray[np.float64] = pickle.load(file)
#
#    with bz2.BZ2File("cnp_lte.pkl", "r") as file:
#    #with open("np_lte.pkl", "rb") as file:
#        lte: npt.NDArray[np.float64] = pickle.load(file)
#    
#    return state_vectors, vectors, lte


def save_data():
    """Saves the numpy arrays."""
    state_vectors, vectors, lte = read_data()

    #with open('state_vectors.npy', 'wb') as file:
    #    np.save(file, state_vectors, allow_pickle = False)

    #with open('vectors.npy', 'wb') as file:
    #    np.save(file, vectors, allow_pickle = False)

    #with open('lte.npy', 'wb') as file:
    #    np.save(file, lte, allow_pickle = False)

    print(lte.shape)
    print(lte.dtype)

    np.save("state_vectors.npy", state_vectors)
    np.save("vectors.npy", vectors)
    np.save("lte.npy", lte)


def load_data() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Loads the numpy arrays."""
    #with open('state_vectors.npy', 'rb') as file:
    #    state_vectors: npt.NDArray[np.float64] = np.load(file)

    #with open('vectors.npy', 'rb') as file:
    #    vectors: npt.NDArray[np.float64] = np.load(file)

    #with open('lte.npy', 'rb') as file:
    #    lte: npt.NDArray[np.float64] = np.load(file)
    #
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


def main():
    """Main entry point."""

    #state_vectors, vectors, lte = read_data()
    #pickle_data()
    #pickle_data_compress()
    #state_vectors, vectors, lte = unpickle_data_uncompress()
    #state_vectors, vectors, lte = unpickle_data()

    save_data()
    #pickle_data()

    state_vectors, vectors, lte = load_data()




    print(f"state_vectors: {state_vectors.shape}")
    print(f"vectors: {vectors.shape}")
    print(f"lte: {lte.shape}")

    matplotlib_testing()

    print(lte.shape)
    print(lte[0][0].shape)

    #file = open("pickle_test.pkl", "wb")
    #pickle.dump(lte, file)
    #file.close()

    #file = open("pickle_test.pkl", "rb")
    #lte_new: npt.NDArray[np.float64] = pickle.load(file)
    #file.close()

    #print(type(lte_new))


    zeroes = [i for i in range(1, lte[0][0].shape[0]+1)]
    plt.plot(zeroes, lte[0, 0, :], label = "original")
    altered = lte + 1
    plt.plot(zeroes, altered[0, 0, :], label = "plus one")
    plt.title("aaaaaaaa")
    plt.legend(loc = "best")
    plt.savefig("output.png")
    print(vectors.shape)


if __name__ == "__main__":
    main()

