import pickle
import numpy as np
import numpy.typing as npt
import pandas as pd
import columns as col


def load_data_embedded():
    """Reads and returns embedded data for the specified (or all) cultivar."""
    with open("concat_embedding_LTE_pred.pkl", "rb") as file:
        raw_lte = pickle.load(file)
        #print(raw_lte["Riesling"]["trial_0"]["0"]["10"].shape)
    #18 cultivars, 3 trials, 337 something, 3 lte values, 366 something

    with open("concat_embedding_penul_vectors.pkl", "rb") as file:
        raw_penul = pickle.load(file)
        print(raw_penul["Riesling"]["trial_0"]["0"][0].shape)
    #18 cultivars, 3 trials, 32 something, 366 something, 1024 vector

    with open("concat_embedding_rnn_vectors.pkl", "rb") as file:
        raw_rnn = pickle.load(file)
    #18 cultivars, 3 trials, 32 something, 366 days (start first day of dormancy and includes summer), 2048 vector

    lte = dict()
    penul = dict()
    rnn = dict()

    # lte 
    for i in raw_lte.keys(): #cultivars
        cur = np.zeros((3, 337, 3, 366), dtype = np.float32)
        cur_cultivar_data = raw_lte[i]
        for j in range(3): #trials
            for k in range(337): #something
                for l in range(3): #lte value
                    cur[j][k][l] = cur_cultivar_data[f"trial_{j}"][str(k)][str(l*40+10)]
        lte[i] = cur

    # penul
    for i in raw_penul.keys(): #cultivars
        cur = np.zeros((3, 32, 366, 1024), dtype = np.float32)
        cur_cultivar_data = raw_penul[i]
        for j in range(3): #trials
            for k in range(32): #seasons, DIFFER BETWEEN CULTIVARS
                cur[j][k] = cur_cultivar_data[f"trial_{j}"][str(k)][0]
        penul[i] = cur

    # rnn
    for i in raw_rnn.keys(): #cultivars
        cur = np.zeros((3, 32, 366, 2048), dtype = np.float32)
        cur_cultivar_data = raw_rnn[i]
        for j in range(3): #trials
            for k in range(32): #seasons
                cur[j][k] = cur_cultivar_data[f"trial_{j}"][str(k)][0]
        rnn[i] = cur





    return lte, penul, rnn


def old_load_data_original() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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


def old_convert_data_numpy():
    """Saves the numpy arrays."""
    state_vectors, vectors, lte = old_load_data_original()

    np.save("state_vectors.npy", state_vectors)
    np.save("vectors.npy", vectors)
    np.save("lte.npy", lte)


def old_load_data_numpy() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Loads the numpy arrays."""
    state_vectors: npt.NDArray[np.float32] = np.load("state_vectors.npy")
    vectors: npt.NDArray[np.float32] = np.load("vectors.npy")
    lte: npt.NDArray[np.float32] = np.load("lte.npy")

    return state_vectors, vectors, lte


def old_convert_data_interoperable():
    """Saves the data as a raw binary file, good for language interoperability."""
    state_vectors, vectors, lte = old_load_data_original()
    state_vectors.tofile("state_vectors.data")
    vectors.tofile("vectors.data")
    lte.tofile("lte.data")


# TODO update this function so you can retrieve the phenology for any cultivar
def get_phenology_dataframe() -> pd.DataFrame:
    """Returns the phenology dataframe with dormancy days"""
    df_raw = pd.read_csv('ColdHardiness_Grape_Prosser_Riesling.csv', sep=',')

    # filter out seasons we don't have RNN data on
    df: pd.DataFrame = df_raw[df_raw[col.SEASON] != "1988-1989"] #filter out 1988
    df = df[df[col.SEASON] != "2001-2002"] #filter out 2001
    df = df.reset_index()
    df[col.DORMANT_DAY] = [0] * len(df[col.DATE]) #prepare column for dormancy days

    # number each day of dormancy in each season
    temp_season = ""
    incr = 0
    for i, row in enumerate(df.iterrows()):
        if row[1][col.DORMANT_SEASON] == 0: #not in a dormant season
            incr = 0
            df[col.DORMANT_DAY][i] = -1
        else: #in a dormant season
            if row[1][col.SEASON] != temp_season: #first day of the dormant season
                temp_season = row[1][col.SEASON]
                incr = 0
                df[col.DORMANT_DAY][i] = 0
            else: #sometime during the dormant season
                incr += 1
                df[col.DORMANT_DAY][i] = incr

    return df



