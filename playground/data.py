import pickle
import numpy as np
import numpy.typing as npt
import pandas as pd
import columns as col
from typing import Any


def load_data_embedded_wrapper() -> tuple[dict[str, Any], dict[str, Any]]:
    """Reads and returns embedded data."""
    #18 cultivars, 3 trials, x seasons, 366 days (starts on first day of dormancy), 2048 vector
    with open("inputs/seasons_parsed.txt", "r") as file:
        cultivar_seasons = file.read().splitlines()
    rnn_dict = dict()
    penul_dict = dict()
    cultivars = []

    for i in range(len(cultivar_seasons) // 5): #for each cultivar
        cultivar_name = cultivar_seasons[i*5]
        trial0 = list(map(lambda x: int(x), cultivar_seasons[i*5+1].split()))
        trial1 = list(map(lambda x: int(x), cultivar_seasons[i*5+2].split()))
        trial2 = list(map(lambda x: int(x), cultivar_seasons[i*5+3].split()))
        cultivars.append((cultivar_name, [trial0, trial1, trial2]))

    with open("inputs/concat_embedding_rnn_vectors.pkl", "rb") as file:
        raw_rnn = pickle.load(file)
    rnn_dict = load_data_embedded_internal(cultivars, raw_rnn, 2048)
    del raw_rnn

    with open("inputs/concat_embedding_penul_vectors.pkl", "rb") as file:
        raw_penul = pickle.load(file)
    penul_dict = load_data_embedded_internal(cultivars, raw_penul, 1024)
    del raw_penul

    return (penul_dict, rnn_dict)


def load_data_embedded_internal(cultivars: list[tuple[str, list[list[int]]]], raw: Any, size: int) -> dict[str, Any]:
    """Reads and returns embedded data, with support from the wrapper function."""
    #18 cultivars, 3 trials, x seasons, 366 days (starts on first day of dormancy), 2048 or 1024 vector
    ret = dict()
    for i in range(len(cultivars)): #for each cultivar
        cultivar = cultivars[i]
        cultivar_name_og = cultivar[0].replace("_", " ")
        count = len(cultivar[1][0]) #number of seasons for this cultivar
        data = np.zeros((3, count, 366, size))
        #18 cultivars, 3 trials, x seasons, 3 lte values, 366 days (starts on first day of dormancy), 2048 vector

        for i, t in enumerate(["trial_0", "trial_1", "trial_2"]): #for each trial
            for s in range(count): #for each of this cultivar's seasons, MAKE SURE THIS IS DONE IN THE RIGHT ORDER
                for d in range(366): #for each day of the season
                    data[i, s, d] = raw[cultivar_name_og][t][f"{cultivar[1][i][s]}"][0][d] #the np array has a 1-width dimension for some reason

        raw[cultivar_name_og] = None #reduce memory usage by deallocating what we no longer need
        ret[cultivar[0]] = data #save data into the return dictionary
    return ret


def old_load_data_original() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Reads and returns data from the pickle files."""

    with open("inputs/state_vectors.pkl", 'rb') as file:
        raw_state_vectors = pickle.load(file)["Riesling"]["trial_0"]  #double nested dictionary
    state_vectors = np.zeros((32, 252, 2048), dtype = np.float32)
    for i in range(32):
        state_vectors[i] = raw_state_vectors[str(i)]  #numpy seems to automatically remove redundant 1D column, [0, :, :]

    with open("inputs/vectors.pkl", 'rb') as file:
        raw_vectors = pickle.load(file)["Riesling"]["trial_0"]
    vectors = np.zeros((32, 252, 1024), dtype = np.float32)
    for i in range(32):
        vectors[i] = raw_vectors[str(i)]

    with open("inputs/LTE.pkl", 'rb') as file:
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

    np.save("inputs/state_vectors.npy", state_vectors)
    np.save("inputs/vectors.npy", vectors)
    np.save("inputs/lte.npy", lte)


def old_load_data_numpy() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Loads the numpy arrays."""
    state_vectors: npt.NDArray[np.float32] = np.load("inputs/state_vectors.npy")
    vectors: npt.NDArray[np.float32] = np.load("inputs/vectors.npy")
    lte: npt.NDArray[np.float32] = np.load("inputs/lte.npy")

    return state_vectors, vectors, lte


def old_convert_data_interoperable():
    """Saves the data as a raw binary file, good for language interoperability."""
    state_vectors, vectors, lte = old_load_data_original()
    state_vectors.tofile("inputs/state_vectors.data")
    vectors.tofile("inputs/vectors.data")
    lte.tofile("inputs/lte.data")


# TODO update this function so you can retrieve the phenology for any cultivar
def get_phenology_dataframe() -> pd.DataFrame:
    """Returns the phenology dataframe with dormancy days"""
    df_raw = pd.read_csv('inputs/ColdHardiness_Grape_Prosser_Riesling.csv', sep=',')

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



