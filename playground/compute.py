import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.cluster

import data


def similarity(seasons: range) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute L2 distance and cosine similarity, returns sizes 249 and 249"""
    state_vectors, _, _ = data.load_data_numpy()
    ret_l2 = np.zeros((len(seasons), 249), dtype = np.float32)
    ret_cos = np.zeros((len(seasons), 249), dtype = np.float32)

    # iterate through all 10 seasons
    for s in seasons:
        cur_vecs: npt.NDArray[np.float32] = state_vectors[s][1:-1] #exclude first and last days
        for i in range(249): #for all elements (except the last one)
            ret_l2[s-seasons[0]][i] = np.linalg.norm((cur_vecs[i]-cur_vecs[i+1]))
            ret_cos[s-seasons[0]][i] = np.dot(cur_vecs[i], cur_vecs[i+1]) / np.linalg.norm(cur_vecs[i]) * np.linalg.norm(cur_vecs[i+1])
    return (ret_l2, ret_cos)


def dbscan(seasons: range) -> npt.NDArray[np.int32]:
    """Runs DBSCAN, returns size 250"""
    state_vectors, _, _ = data.load_data_numpy()
    ret = np.zeros((len(seasons), 250), dtype = np.int32)

    eps = 3.0 #maximum distance for two points to be in the same neighborhood
    min_samples = 4 #samples required to be considered a core point

    model = sklearn.cluster.DBSCAN(
            eps = eps,
            min_samples = min_samples,
            )

    for s in seasons: #cover each season
        cur_states = state_vectors[s][1:-1] #grab state vectors for just this season
        fitted = model.fit(cur_states) #need to fit to X, must find what X to give it
        ret[s-seasons[0]] = fitted.labels_

    return ret


def k_span(k: int, seasons: range) -> npt.NDArray[np.float32]:
    """Runs Ananth's K-scan, returns size 250-k"""
    state_vectors, _, _ = data.load_data_numpy()
    ret = np.zeros((len(seasons), 250-k), dtype = np.float32)

    # for each season
    for s in seasons: #cover each season
        cur_states = state_vectors[s][1:-1] #grab state vectors for just this season

        # for each day
        for i in range(250 - k): #don't overshoot the number of comparisons
            points = cur_states[i:i+k+1] #all the points we'll be comparing
            max = 0

            #find the max distance between the points
            for p in range(k+1): #for each point
                for q in range(k+1): #for every other point
                    if p == q: #skip when we're comparing a day to itself
                        continue
                    else:
                        dist = np.linalg.norm(points[p]-points[q])
                        if dist > max:
                            max = dist

            ret[s-seasons[0]][i] = max

    return ret





