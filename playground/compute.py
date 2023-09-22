import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd

import sklearn.cluster

import data
import columns as col









def similarity() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute L2 distance and cosine similarity, returns sizes 249 and 249"""
    state_vectors, _, _ = data.load_data_numpy()
    #phenology_df = data.get_phenology_dataframe()
    ret_l2 = np.zeros((10, 249), dtype = np.float32)
    ret_cos = np.zeros((10, 249), dtype = np.float32)

    '''
    Notes:

    state_vectors has only values between -1 and 1
    with 2048 dimensions, this means a vector can be a maximum of 45.254833995939045 in length (sqrt of 2048)

    want to output graphs for seasons 2002-2003 through 2011-2012, which are indices 10 through 19

    want to compute different clusters of points, and these clusters will color the points on the graph
        - a cluster could be revisted later theoretically, so it will have two+ areas of the graph of the same color

    consider splitting data module into subtree of io, functions, etc

    '''

    # iterate through all 10 seasons
    for s in range(10, 20):
        cur_vecs: npt.NDArray[np.float32] = state_vectors[s][1:-1] #exclude first and last days
        #cur_season = seasons[s-10]

        # comparisons
        #euclidean_distances = np.zeros(249)
        #cosine_similarities = np.zeros(249)
        for i in range(249):
            #euclidean_distances[i] = np.linalg.norm((cur_vecs[i]-cur_vecs[i+1]))
            #cosine_similarities[i] = np.dot(cur_vecs[i], cur_vecs[i+1]) / np.linalg.norm(cur_vecs[i]) * np.linalg.norm(cur_vecs[i+1])
            ret_l2[s-10][i] = np.linalg.norm((cur_vecs[i]-cur_vecs[i+1]))
            ret_cos[s-10][i] = np.dot(cur_vecs[i], cur_vecs[i+1]) / np.linalg.norm(cur_vecs[i]) * np.linalg.norm(cur_vecs[i+1])

        # phenology data
        #cur_phenologies = phenology_for_season(phenology_df, cur_season)

        # output graph euclidean / l2
        #plt.close()
        #plt.clf()
        #plt.figure(figsize = (6.4, 4.8), dpi = 100)
        #plt.plot(list(range(1, 250)), euclidean_distances)
        #plt.title("Euclidean Distances " + cur_season)
        #insert_phenology(cur_phenologies, 0.5)
        #plt.savefig("output_graphs/euclidean_" + cur_season + ".png")

        ## output graph cosine similarity
        #plt.close()
        #plt.clf()
        #plt.figure(figsize = (6.4, 4.8), dpi = 100)
        #plt.plot(list(range(1, 250)), cosine_similarities)
        #plt.title("Cosine Similarities " + cur_season)
        #insert_phenology(cur_phenologies, 1450)
        #plt.savefig("output_graphs/cosine_" + cur_season + ".png")

    return (ret_l2, ret_cos)


def dbscan() -> npt.NDArray[np.int32]:
    """Runs DBSCAN, returns size 250"""
    state_vectors, _, _ = data.load_data_numpy()
    #phenology_df = data.get_phenology_dataframe()
    ret = np.zeros((10, 250), dtype = np.int32)

    eps = 3.0 #maximum distance for two points to be in the same neighborhood
    min_samples = 4 #samples required to be considered a core point

    model = sklearn.cluster.DBSCAN(
            eps = eps,
            min_samples = min_samples,
            )

    for s in range(10, 20): #cover each season
        #cur_season = seasons[i-10]

        cur_states = state_vectors[s][1:-1] #grab state vectors for just this season
        fitted = model.fit(cur_states) #need to fit to X, must find what X to give it
        #print(type(fitted.labels_))
        #print(fitted.labels_.shape)
        #print(fitted.labels_)
        #print(fitted.labels_.shape)
        #print(np.max(fitted.labels_))

        # phenology data
        #cur_phenologies = phenology_for_season(phenology_df, cur_season)

        # output graph dbscan
        #plt.close()
        #plt.clf()
        #plt.figure(figsize = (6.4, 4.8), dpi = 100)
        #plt.plot(list(range(1,251)), fitted.labels_)
        ##plt.title(f"DBSCAN {cur_season} eps={eps}, min_samples={min_samples}")
        #plt.title(f"DBSCAN {cur_season}")
        #insert_phenology(cur_phenologies, 0.5)
        #plt.savefig("output_graphs/dbscan_" + cur_season + ".png")

        ret[s-10] = fitted.labels_

    #print(state_vectors.shape)
    return ret


def k_span(k: int) -> npt.NDArray[np.float32]:
    """Runs Ananth's K-scan, returns size 250-k"""
    state_vectors, _, _ = data.load_data_numpy()
    #phenology_df = data.get_phenology_dataframe()
    ret = np.zeros((10, 250-k), dtype = np.float32)

    #distances = np.zeros(250 - k)

    # for each season
    for s in range(10, 20): #cover each season
        #cur_season = seasons[s-10]
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
                        #euclidean_distances[i] = np.linalg.norm((cur_vecs[i]-cur_vecs[i+1]))
                        if dist > max:
                            max = dist

            ret[s-10][i] = max

        # phenology data
        #cur_phenologies = phenology_for_season(phenology_df, cur_season)

        # output graph k-span
        #plt.close()
        #plt.clf()
        #plt.figure(figsize = (6.4, 4.8), dpi = 100)
        #plt.plot(list(range(1,251-k)), distances)
        ##plt.title(f"DBSCAN {cur_season} eps={eps}, min_samples={min_samples}")
        #plt.title(f"K-span({k}) {cur_season}")
        #insert_phenology(cur_phenologies, 3)
        #plt.savefig(f"output_graphs/kspan{k}_" + cur_season + ".png")

    return ret








def similarity_graph():
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

    # iterate through all 10 seasons
    for i in range(10, 20):
        cur_vecs: npt.NDArray[np.float32] = state_vectors[i][1:-1] #exclude first and last days
        cur_season = seasons[i-10]

        # comparisons
        euclidean_distances = np.zeros(249)
        cosine_similarities = np.zeros(249)
        for i in range(249):
            euclidean_distances[i] = np.linalg.norm((cur_vecs[i]-cur_vecs[i+1]))
            cosine_similarities[i] = np.dot(cur_vecs[i], cur_vecs[i+1]) / np.linalg.norm(cur_vecs[i]) * np.linalg.norm(cur_vecs[i+1])

        # phenology data
        cur_phenologies = phenology_for_season(phenology_df, cur_season)

        # output graph euclidean / l2
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1, 250)), euclidean_distances)
        plt.title("Euclidean Distances " + cur_season)
        insert_phenology(cur_phenologies, 0.5)
        plt.savefig("output_graphs/euclidean_" + cur_season + ".png")

        # output graph cosine similarity
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1, 250)), cosine_similarities)
        plt.title("Cosine Similarities " + cur_season)
        insert_phenology(cur_phenologies, 1450)
        plt.savefig("output_graphs/cosine_" + cur_season + ".png")


def dbscan_graph():
    state_vectors, _, _ = data.load_data_numpy()
    phenology_df = data.get_phenology_dataframe()

    eps = 3.0 #maximum distance for two points to be in the same neighborhood
    min_samples = 4 #samples required to be considered a core point

    model = sklearn.cluster.DBSCAN(
            eps = eps,
            min_samples = min_samples,
            )

    for i in range(10, 20): #cover each season
        cur_season = seasons[i-10]

        cur_states = state_vectors[i][1:-1] #grab state vectors for just this season
        fitted = model.fit(cur_states) #need to fit to X, must find what X to give it
        #print(fitted.labels_)
        #print(fitted.labels_.shape)
        #print(np.max(fitted.labels_))

        # phenology data
        cur_phenologies = phenology_for_season(phenology_df, cur_season)

        # output graph dbscan
        plt.close()
        plt.clf()
        plt.figure(figsize = (6.4, 4.8), dpi = 100)
        plt.plot(list(range(1,251)), fitted.labels_)
        #plt.title(f"DBSCAN {cur_season} eps={eps}, min_samples={min_samples}")
        plt.title(f"DBSCAN {cur_season}")
        insert_phenology(cur_phenologies, 0.5)
        plt.savefig("output_graphs/dbscan_" + cur_season + ".png")

    print(state_vectors.shape)


def k_span_graph():
    state_vectors, _, _ = data.load_data_numpy()
    phenology_df = data.get_phenology_dataframe()

    # for each group size
    for k in [5, 10]: #the max number of points we'll look forward
        distances = np.zeros(250 - k)

        # for each season
        for s in range(10, 20): #cover each season
            cur_season = seasons[s-10]
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
                            #euclidean_distances[i] = np.linalg.norm((cur_vecs[i]-cur_vecs[i+1]))
                            if dist > max:
                                max = dist

                distances[i] = max

            # phenology data
            cur_phenologies = phenology_for_season(phenology_df, cur_season)

            # output graph k-span
            plt.close()
            plt.clf()
            plt.figure(figsize = (6.4, 4.8), dpi = 100)
            plt.plot(list(range(1,251-k)), distances)
            #plt.title(f"DBSCAN {cur_season} eps={eps}, min_samples={min_samples}")
            plt.title(f"K-span({k}) {cur_season}")
            insert_phenology(cur_phenologies, 3)
            plt.savefig(f"output_graphs/kspan{k}_" + cur_season + ".png")

            

        









