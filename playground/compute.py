import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.cluster

import data


def dbscan(seasons: range, eps: float, min_samples: int) -> npt.NDArray[np.int32]:
    """Runs DBSCAN, returns size (seasons, 250)"""
    state_vectors, _, _ = data.load_data_numpy()
    ret = np.zeros((len(seasons), 250), dtype = np.int32)

    model = sklearn.cluster.DBSCAN(
            eps = eps, # maximum distance for two points to be in the same neighborhood
            min_samples = min_samples, # samples required to be considered a core point
            )

    for s in seasons: #cover each season
        cur_states = state_vectors[s][1:-1] #grab state vectors for just this season
        fitted = model.fit(cur_states) #need to fit to X, must find what X to give it
        ret[s-seasons[0]] = fitted.labels_

    return ret


def k_span(seasons: range, k: int) -> npt.NDArray[np.float32]:
    """Runs Ananth's K-span, returns size (seasons, 250-k)"""
    state_vectors, _, _ = data.load_data_numpy()
    ret = np.zeros((len(seasons), 250-k), dtype = np.float32)

    # for each season
    for s in seasons: #cover each season
        cur_states = state_vectors[s][1:-1] #grab state vectors for just this season

        # for each day
        for i in range(250 - k): #don't overshoot the number of comparisons
            points = cur_states[i:i+k+1] #all the points we'll be comparing
            avg = 0

            #find the max distance between the points
            for p in range(k+1): #for each point
                for q in range(k+1): #for every other point
                    if p == q: #skip when we're comparing a day to itself
                        continue
                    else:
                        dist = np.linalg.norm(points[p]-points[q])
                        avg += dist

            ret[s-seasons[0]][i] = avg / (250 - k) 

    return ret


def cluster_stats(seasons: range, clusters: npt.NDArray[np.int32]) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]]:
    """Finds the cluster's stats"""
    state_vectors, _, _ = data.load_data_numpy()
    clusters_max = [None]*len(seasons)
    clusters_avg = [None]*len(seasons)

    # for each season
    for i, s in enumerate(seasons): #cover each season
        cur_states = state_vectors[s][1:-1] #grab state vectors for just this season
        cur_clusters = clusters[i] #grab cluster labels for points
        num_clusters = np.max(cur_clusters) + 2 #includes outliers

        cur_max = np.zeros(num_clusters)
        cur_avg = np.zeros(num_clusters)
        cur_num = np.zeros(num_clusters) #keep track of size of clusters

        #find the stats in a cluster
        for p in range(250): #for each point
            cur_cluster = cur_clusters[p]
            cur_num[cur_cluster] += 1 #increment the count
            for q in range(250): #for every other point
                if p == q: #skip when we're comparing a day to itself
                    continue
                if cur_clusters[p] != cur_clusters[q]: #skip when the cluster doesn't match
                    continue
                dist = np.linalg.norm(cur_states[p]-cur_states[q]) #get distance
                if cur_max[cur_cluster] < dist: #update the new max distance
                    cur_max[cur_cluster] = dist
                cur_avg[cur_cluster] += dist

        for c in range(num_clusters):
            cur_avg[c] = cur_avg[c] / cur_num[c] #divide the sum by the num of points

        # store the current stats into the return values
        clusters_max[i] = cur_max
        clusters_avg[i] = cur_avg

    return clusters_max, clusters_avg




#def similarity(seasons: range) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
#    """Compute L2 distance and cosine similarity, returns sizes 249 and 249"""
#    state_vectors, _, _ = data.load_data_numpy()
#    ret_l2 = np.zeros((len(seasons), 249), dtype = np.float32)
#    ret_cos = np.zeros((len(seasons), 249), dtype = np.float32)
#
#    # iterate through all 10 seasons
#    for s in seasons:
#        cur_vecs: npt.NDArray[np.float32] = state_vectors[s][1:-1] #exclude first and last days
#        for i in range(249): #for all elements (except the last one)
#            ret_l2[s-seasons[0]][i] = np.linalg.norm((cur_vecs[i]-cur_vecs[i+1]))
#            ret_cos[s-seasons[0]][i] = np.dot(cur_vecs[i], cur_vecs[i+1]) / np.linalg.norm(cur_vecs[i]) * np.linalg.norm(cur_vecs[i+1])
#    return (ret_l2, ret_cos)


