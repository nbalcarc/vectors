import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

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
    deltas = np.zeros(250)

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
    for i in range(250):
        deltas[i] = cosine_similarities[i+1] - cosine_similarities[i]

    print("Euclidean distances:")
    #print(euclidean_distances)
    for i in range(251):
        print(f"{i}: {euclidean_distances[i]}")

    print("Cosine similarity:")
    #print(cosine_similarities)
    for i in range(251):
        print(f"{i}: {cosine_similarities[i]}")

    print("Cosine deltas:")
    for i in range(250):
        print(f"{i}: {deltas[i]}")

    #print("Angles:")
    #print(angle_mags)


    '''
    TODO:
    graph the euclidean distances and see if there's any patterns for where the peaks and flatlands are
    find the delta between the cosine similarities and see if there's any drops or rises
    '''

    plt.clf()
    plt.plot(list(range(251)), euclidean_distances)
    plt.title("Euclidean Distances")
    plt.savefig("euclidean.png")

    plt.clf()
    plt.plot(list(range(251)), cosine_similarities)
    plt.title("Cosine Similarities")
    plt.savefig("cosine.png")

    plt.clf()
    plt.plot(list(range(250)), deltas)
    plt.title("Cosine Deltas")
    plt.savefig("deltas.png")




