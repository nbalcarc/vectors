import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd

import data
import columns as col


def similarity():
    state_vectors, _, _ = data.load_data()
    df_raw = data.load_phenology_csv()

    #NOTE: in the RNN data, years 1988-1989 and 2001-2002 are excluded, which explains the size difference
    df: pd.DataFrame = df_raw[df_raw[col.SEASON] != "1988-1989"] #filter out 1988
    df = df[df[col.SEASON] != "2001-2002"] #filter out 2001
    df = df.reset_index()
    df[col.DORMANT_DAY] = [0] * len(df[col.DATE])

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

    #df.to_csv("output.csv") //output file with numbered dormancy days

    # grab all the rows with phenology data
    df_temp = df.copy()
    df_temp[col.PHENOLOGY].fillna(0, inplace=True)
    phenology = df_temp[df_temp[col.PHENOLOGY] != 0]
    phenology = phenology[phenology[col.DORMANT_SEASON] == 1] #filter to only within the dormancy season

    '''
    Notes:

    state_vectors has only values between -1 and 1
    with 2048 dimensions, this means a vector can be a maximum of 45.254833995939045 in length (sqrt of 2048)

    want to output graphs for seasons 2002-2003 through 2011-2012, which are indices 10 through 19

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
        cur_phenologies: pd.DataFrame = phenology[phenology[col.SEASON] == cur_season] #filter to only the current season
        cur_phenologies = cur_phenologies[[col.PHENOLOGY, col.DORMANT_DAY]] #only care about phenology and the day

        # simplifies the process of printing phenology data
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

        # euclidean / l2
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1, 250)), euclidean_distances[1:-1]) #skip the first and last indices
        plt.title("Euclidean Distances")
        insert_phenology(0.5)
        plt.savefig("output_graphs/euclidean_" + cur_season + ".png")

        # cosine similarity
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1, 250)), cosine_similarities[1:-1])
        plt.title("Cosine Similarities")
        insert_phenology(1450)
        plt.savefig("output_graphs/cosine_" + cur_season + ".png")





