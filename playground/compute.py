import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd

import sklearn.cluster

import data
import columns as col




def phenology_for_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Grab the phenology data for the current season"""
    cur: pd.DataFrame = df[df[col.SEASON] == season] #filter to only the current season
    dorm = cur[cur[col.DORMANT_SEASON] == 1].copy() #filter to only within the dormancy season
    dorm[col.PHENOLOGY].fillna(0, inplace=True)
    clean = dorm[dorm[col.PHENOLOGY] != 0] #filter out NaNs
    short: pd.DataFrame = clean[[col.PHENOLOGY, col.DORMANT_DAY]] #only care about phenology and the day

    return short


def similarity():
    """Compute L2 distance and cosine similarity"""
    state_vectors, _, _ = data.load_data_numpy()
    phenology_df = data.get_phenology_dataframe()

    '''
    Notes:

    state_vectors has only values between -1 and 1
    with 2048 dimensions, this means a vector can be a maximum of 45.254833995939045 in length (sqrt of 2048)

    want to output graphs for seasons 2002-2003 through 2011-2012, which are indices 10 through 19

    want to compute different clusters of points, and these clusters will color the points on the graph
        - a cluster could be revisted later theoretically, so it will have two+ areas of the graph of the same color

    consider splitting data module into subtree of io, functions, etc

    '''

    seasons = [
            "2002-2003",
            "2003-2004",
            "2004-2005",
            "2005-2006",
            "2006-2007",
            "2007-2008",
            "2008-2009",
            "2009-2010",
            "2010-2011",
            "2011-2012",
            ]

    # iterate through all 10 seasons
    for i in range(10, 20):
        cur_vecs: npt.NDArray[np.float32] = state_vectors[i]
        cur_season = seasons[i-10]

        # phenology data
        cur_phenologies = phenology_for_season(phenology_df, cur_season)

        # adds phenology data to the graph, asks for a y-coordinate of the labels
        def insert_phenology(y_coordinate: float):
            for row in cur_phenologies.iterrows():
                plt.axvline(row[1][1], color = "red") #graph at the specified index
                plt.text(row[1][1], y_coordinate, row[1][0], rotation=90) #add a phenology label

        # comparisons
        euclidean_distances = np.zeros(251)
        cosine_similarities = np.zeros(251)
        for i in range(251):
            euclidean_distances[i] = np.linalg.norm((cur_vecs[i]-cur_vecs[i+1]))
            cosine_similarities[i] = np.dot(cur_vecs[i], cur_vecs[i+1]) / np.linalg.norm(cur_vecs[i]) * np.linalg.norm(cur_vecs[i+1])

        # output graph euclidean / l2
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1, 250)), euclidean_distances[1:-1]) #skip the first and last indices
        plt.title("Euclidean Distances")
        insert_phenology(0.5)
        plt.savefig("output_graphs/euclidean_" + cur_season + ".png")

        # output graph cosine similarity
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1, 250)), cosine_similarities[1:-1])
        plt.title("Cosine Similarities")
        insert_phenology(1450)
        plt.savefig("output_graphs/cosine_" + cur_season + ".png")


def dbscan():
    state_vectors, _, _ = data.load_data_numpy()

    model = sklearn.cluster.DBSCAN(
            eps = 3.0, #maximum distance for two points to be in the same neighborhood
            min_samples = 4,#samples required to be considered a core point
            )

    for i in range(10, 20): #cover each season
        cur_states = state_vectors[i] #grab state vectors for just this season
        fitted = model.fit(cur_states) #need to fit to X, must find what X to give it
        print(fitted.labels_)
        print(fitted.labels_.shape)
        print(np.max(fitted.labels_))

    print(state_vectors.shape)







