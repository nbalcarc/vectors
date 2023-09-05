import numpy as np
import numpy.typing as npt
import math

import data

def similarity():
    state_vectors, _, _ = data.load_data()

    '''
    Notes:

    state_vectors has only values between -1 and 1
    with 2048 dimensions, this means a vector can be a maximum of 45.254833995939045 in length (sqrt of 2048)

    '''

    cur_vecs: npt.NDArray[np.float32] = state_vectors[0]

    angle_mags = np.zeros((252, 2048))
    euclidean_distances = np.zeros(251)
    cosine_similarities = np.zeros(251)

    # individual vector processing
    for i in range(252):
        angle_mag = np.zeros(2048)
        for j in range(2047): #want to leave the final index empty, that's where the magnitude goes
            angle_mag[j] = math.atan2(cur_vecs[i][j], cur_vecs[i][j+1])
        angle_mag[2047] = np.linalg.norm(cur_vecs[i])
        angle_mags[i] = angle_mag

    # comparative vector processing
    for i in range(251):
        euclidean_distances[i] = np.linalg.norm((cur_vecs[i]-cur_vecs[i+1]))
        cosine_similarities[i] = np.dot(cur_vecs[i], cur_vecs[i+1]) / np.linalg.norm(cur_vecs[i]) * np.linalg.norm(cur_vecs[i+1])

    print("Euclidean distances:")
    print(euclidean_distances)

    print("Cosine similarity:")
    print(cosine_similarities)

    #lengths = np.array(list(map(np.linalg.norm, cur_vecs)))
    #min = lengths.min()
    #max = lengths.max()

    #print("Raw:")
    #print(cur_vecs)

    #print(f"Min length: {min}")
    #print(f"Max length: {max}")

    #angles = np.array(list(map(np.arctan, cur_vecs)))
    print("Angles:")
    print(angle_mags)




