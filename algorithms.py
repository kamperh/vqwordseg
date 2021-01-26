"""
VQ phone and word segmentation algorithms.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from scipy.spatial import distance
import numpy as np


#-----------------------------------------------------------------------------#
#                  DYNAMIC PROGRAMMING PENALIZED SEGMENTATION                 #
#-----------------------------------------------------------------------------#

def get_segment_intervals(n_total, n_max_frames):
    indices = [None]*int((n_total**2 + n_total)/2)
    for cur_start in range(n_total):
        for cur_end in range(cur_start, min(n_total, cur_start +
                n_max_frames)):
            cur_end += 1
            t = cur_end
            i = int(t*(t - 1)/2)
            indices[i + cur_start] = (cur_start, cur_end)
    return indices


def custom_viterbi(costs, n_frames):
    """
    Viterbi segmentation of an utterance of length `n_frames` based on `costs`.

    Parameters
    ----------
    costs : n_frames(n_frames + 1)/2 vector
        For t = 1, 2, ..., N the entries costs[i:i + t] contains the costs of
        seq[0:t] up to seq[t - 1:t], with i = t(t - 1)/2. Written out: costs =
        [cost(seq[0:1]), cost(seq[0:2]), cost(seq[1:2]), cost(seq[0:3]), ...,
        cost(seq[N-1:N])].

    Return
    ------
    (summed_cost, boundaries) : (float, vector of bool)
    """
    
    # Initialise
    boundaries = np.zeros(n_frames, dtype=bool)
    boundaries[-1] = True
    alphas = np.ones(n_frames)
    alphas[0] = 0.0

    # Forward filtering
    i = 0
    for t in range(1, n_frames):
        alphas[t] = np.min(
            costs[i:i + t] + alphas[:t]
            )
        i += t

    # print("alphas: {}".format(alphas))

    # Backward segmentation
    t = n_frames
    summed_cost = 0.0
    while True:
        i = int(0.5*(t - 1)*t)
        q_t_min_list = (
            costs[i:i + t] + alphas[:t]       
            )
        q_t_min_list = q_t_min_list[::-1]
        q_t = np.argmin(q_t_min_list) + 1

        # print("-"*39)
        # print("t = {}".format(t))
        # print("q_t_min_list: {}".format(q_t_min_list))
        # print("arg min: {}".format(q_t))
        # print("Cost: {:.4f}".format(costs[i + t - q_t]))

        summed_cost += costs[i + t - q_t]
        if t - q_t - 1 < 0:
            break
        boundaries[t - q_t - 1] = True
        t = t - q_t

    # print("Utterance loss: {:.4f}".format(summed_cost))
    return summed_cost, boundaries


def dp_penalized(embedding, z, n_min_frames=0, n_max_frames=15,
        dur_weight=20**2):

    # Hyperparameters
    count_weight = 0
       
    # Distances between each z and each embedding (squared Euclidean)
    embedding_distances = distance.cdist(z, embedding, metric="sqeuclidean")
    # print("embedding_distances shape: {}".format(embedding_distances.shape))
    
    # Costs for segment intervals
    segment_intervals = get_segment_intervals(z.shape[0], n_max_frames)
    costs = np.inf*np.ones(len(segment_intervals))
    for i_seg, interval in enumerate(segment_intervals):
        if interval is None:
            continue
        i_start, i_end = interval
        dur = i_end - i_start
        if dur < n_min_frames:
            continue
        cost = np.min(
            np.sum(embedding_distances[i_start:i_end, :], axis=0)
            ) - dur_weight*(dur - 1) + count_weight
        costs[i_seg] = cost
    
    # Viterbi segmentation
    summed_cost, boundaries = custom_viterbi(costs, z.shape[0])
    
    # Code assignments
    segmented_codes = []
    j_prev = 0
    for j in np.where(boundaries)[0]:
        i_start = j_prev
        i_end = j + 1
        code = np.argmin(np.sum(embedding_distances[i_start:i_end, :], axis=0))
        segmented_codes.append((i_start, i_end, code))
        j_prev = j + 1
    
    return boundaries, segmented_codes


#-----------------------------------------------------------------------------#
#        N-SEG. CONSTRAINED DYNAMIC PROGRAMMING PENALIZED SEGMENTATION        #
#-----------------------------------------------------------------------------#

def custom_viterbi_n_segments(costs, n_frames, n_segments):
    """
    Viterbi segmentation of an utterance of length `n_frames` based on `costs`
    constrained to produce `n_segments`.

    Parameters
    ----------
    costs : n_frames(n_frames + 1)/2 vector
        For t = 1, 2, ..., N the entries costs[i:i + t] contains the costs of
        seq[0:t] up to seq[t - 1:t], with i = t(t - 1)/2. Written out: costs =
        [cost(seq[0:1]), cost(seq[0:2]), cost(seq[1:2]), cost(seq[0:3]), ...,
        cost(seq[N-1:N])].

    Return
    ------
    (summed_cost, boundaries) : (float, vector of bool)
    """
    
    # Initialise
    boundaries = np.zeros(n_frames, dtype=bool)
    boundaries[-1] = True
    alphas = np.inf*np.ones((n_frames, n_segments + 1))
    alphas[0, 0] = 0.0

    # Forward filtering
    i = 0
    for t in range(1, n_frames):
        for s in range(1, n_segments):
            alphas[t, s] = np.min(
                costs[i:i + t] + alphas[:t, s - 1]
                )  # vectorise (?)
        i += t

    # print("alphas: {}".format(alphas))

    # Backward segmentation
    t = n_frames
    summed_cost = 0.0
    s = n_segments
    while True:
        i = int(0.5*(t - 1)*t)
        q_t_min_list = (
            costs[i:i + t] + alphas[:t, s - 1]
            )
        q_t_min_list = q_t_min_list[::-1]
        q_t = np.argmin(q_t_min_list) + 1

        # print("-"*39)
        # print("t = {}".format(t))
        # print("q_t_min_list: {}".format(q_t_min_list))
        # print("arg min: {}".format(q_t))
        # print("Cost: {:.4f}".format(costs[i + t - q_t]))
        
        summed_cost += costs[i + t - q_t]
        if t - q_t - 1 < 0:
            break
        boundaries[t - q_t - 1] = True
        t = t - q_t
        s -= 1

    # print("Utterance loss: {:.4f}".format(summed_cost))
    return summed_cost, boundaries

def dp_penalized_n_seg(embedding, z, n_min_frames=0, n_max_frames=15,
        dur_weight=0, n_frames_per_segment=7, n_min_segments=0):

    # Hyperparameters
    n_segments = max(1, int(round(z.shape[0]/n_frames_per_segment)))
    if n_segments < n_min_segments:
        n_segments = n_min_segments
    assert n_max_frames*n_segments >= z.shape[0]

    # Distances between each z and each embedding (squared Euclidean)
    embedding_distances = distance.cdist(z, embedding, metric="sqeuclidean")
    
    # Costs for segment intervals
    segment_intervals = get_segment_intervals(z.shape[0], n_max_frames)
    costs = np.inf*np.ones(len(segment_intervals))
    for i_seg, interval in enumerate(segment_intervals):
        if interval is None:
            continue
        i_start, i_end = interval
        dur = i_end - i_start
        if dur < n_min_frames:
            continue
        cost = np.min(
            np.sum(embedding_distances[i_start:i_end, :], axis=0)
            ) - dur_weight*(dur - 1)
        costs[i_seg] = cost
    
    # Viterbi segmentation
    summed_cost, boundaries = custom_viterbi_n_segments(
        costs, z.shape[0], n_segments
        )
    
    # Code assignments
    segmented_codes = []
    j_prev = 0
    for j in np.where(boundaries)[0]:
        i_start = j_prev
        i_end = j + 1
        code = np.argmin(np.sum(embedding_distances[i_start:i_end, :], axis=0))
        segmented_codes.append((i_start, i_end, code))
        j_prev = j + 1
    
    return boundaries, segmented_codes


#-----------------------------------------------------------------------------#
#                         WORD SEGMENTATION ALGORITHMS                        #
#-----------------------------------------------------------------------------#

def ag(utterance_list, nruns=4, njobs=3, args="-n 100"):
    from wordseg.algos import ag
    return list(ag.segment(
        utterance_list, nruns=nruns, njobs=njobs, args=args
        ))


def tp(utterance_list, threshold="relative"):
    from wordseg.algos import tp
    import wordseg.algos
    return list(tp.segment(utterance_list, threshold=threshold))
