import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
import math

import data
import columns as col


def similarity():
    state_vectors, _, _ = data.load_data()
    df_raw = data.load_phenology_csv()
    df: pd.DataFrame = df_raw[df_raw[col.SEASON] != "1988-1989"] #filter out 1988
    df = df[df[col.SEASON] != "2001-2002"] #filter out 2001
    df[41] = [0] * len(df[col.DATE])
    #yy = [0] * len(df[col.DATE])
    #print(len(yy))
    #print(len(df[col.DATE]))

    # number all of the dates in each dormant season
    temp_season = ""
    incr = 0
    #dorm = df[col.DORMANT_SEASON and 41].reset_index()
    dorm = df[[col.DORMANT_SEASON, 41]].reset_index()
    print('sup')
    print(type(dorm))
    print(dorm)
    print(dorm.keys())
    print(dorm[col.DORMANT_SEASON])

    season: pd.Series = df[col.SEASON]

    print("ok done")
    for i in range(0, len(df[col.DATE])):
        print(i)
        if dorm[0][i] == '0': #not in a dormant season
            df[41][i] = -1
        if dorm[0][i] == '1': #in dormant season
            if season[0][i] != temp_season:
                temp_season = season[0][i]
                incr = 0
                df[41][i] = incr
            else:
                incr += 1
                df[41][i] = incr

    dormancy_filtered = df[df[col.DORMANT_SEASON] == '1'] #grab all the rows that are part of the dormancy season

    # seasons are distinguished by the year, so we may want to number all of these
    #x = [x for x in range(0, 252)]
    #y = x * 32
    #print(x)
    #print("hi")
    #print(len(y))
    print(len(dormancy_filtered))
    y = list(set(dormancy_filtered[col.SEASON]))
    #y.sort()
    #print(y)
    #print(len(y))

    #NOTE: in the RNN data, years 1988-1989 and 2001-2002 are excluded, which explains the size difference

    #phenology_filtered = phenology_raw[phenology_raw[col.PHENOLOGY]] #grab all phenology data
    #phenology_filtered = phenology_raw.loc[isinstance(phenology_raw[col.PHENOLOGY], float)] #grab all phenology data
    #phenology_filtered = phenology_raw
    #phenology = df_raw.copy()[col.PHENOLOGY].fillna(0, inplace=True)
    df_raw[col.PHENOLOGY].fillna(0, inplace=True)
    phenology = df_raw[df_raw[col.PHENOLOGY] != 0]
    
    print(phenology[col.PHENOLOGY])

    #print(phenology_filtered)
    #print(phenology_filtered[col.PHENOLOGY])
    #print(phenology_filtered[col.PHENOLOGY][1])
    #print(type(phenology_filtered[col.PHENOLOGY][1]))

    return

    '''
    Notes:

    state_vectors has only values between -1 and 1
    with 2048 dimensions, this means a vector can be a maximum of 45.254833995939045 in length (sqrt of 2048)

    '''

    cur_vecs: npt.NDArray[np.float32] = state_vectors[0]

    angle_mags = np.zeros((252, 2048))
    euclidean_distances = np.zeros(251)
    cosine_similarities = np.zeros(251)
    #deltas = np.zeros(250)

    # individual vector processing
    for i in range(252):
        angle_mag = np.zeros(2048)
        for j in range(2047): #want to leave the final index empty, that's where the magnitude goes
            angle_mag[j] = np.arctan2(cur_vecs[i][j], cur_vecs[i][j+1])
        angle_mag[2047] = np.linalg.norm(cur_vecs[i])
        angle_mags[i] = angle_mag

    # comparative vector processing
    for i in range(251):
        euclidean_distances[i] = np.linalg.norm((cur_vecs[i]-cur_vecs[i+1]))
        cosine_similarities[i] = np.dot(cur_vecs[i], cur_vecs[i+1]) / np.linalg.norm(cur_vecs[i]) * np.linalg.norm(cur_vecs[i+1])

    # further processing on previous loop
    #for i in range(250):
    #    deltas[i] = cosine_similarities[i+1] - cosine_similarities[i]

    #print("Euclidean distances:")
    #print(euclidean_distances)
    #for i in range(251):
    #    print(f"{i}: {euclidean_distances[i]}")

    #print("Cosine similarity:")
    #print(cosine_similarities)
    #for i in range(251):
    #    print(f"{i}: {cosine_similarities[i]}")

    #print("Cosine deltas:")
    #for i in range(250):
    #    print(f"{i}: {deltas[i]}")

    #print("Angles:")
    #print(angle_mags)


    '''
    TODO:
    graph the euclidean distances and see if there's any patterns for where the peaks and flatlands are
    find the delta between the cosine similarities and see if there's any drops or rises
    '''

    plt.clf()
    plt.plot(list(range(1,251)), euclidean_distances[1:])
    plt.title("Euclidean Distances")
    plt.savefig("euclidean.png")

    plt.clf()
    plt.plot(list(range(1,251)), cosine_similarities[1:])
    plt.title("Cosine Similarities")
    plt.savefig("cosine.png")

    #plt.clf()
    #plt.plot(list(range(250)), deltas)
    #plt.title("Cosine Deltas")
    #plt.savefig("deltas.png")




