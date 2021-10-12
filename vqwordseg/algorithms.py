"""
VQ phone and word segmentation algorithms.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from scipy.spatial import distance
from scipy.special import factorial
from scipy.stats import gamma
from tqdm import tqdm
import numpy as np


#-----------------------------------------------------------------------------#
#         PHONE DURATION PRIORS: NEGATIVE LOG PROB (WANT TO MINIMIZE)         #
#-----------------------------------------------------------------------------#

def neg_chorowski(dur, weight=None):
    score = -(dur - 1)
    if weight is None:
        return score
    else:
        return -weight*score


def neg_log_poisson(dur, poisson_param=5, weight=None):
    return -(
        -poisson_param + dur*np.log(poisson_param) - np.log(factorial(dur))
        )


histogram = np.array([
    4.94283846e-05, 7.72517818e-03, 3.58084730e-02, 1.00731859e-01,
    1.14922589e-01, 1.16992203e-01, 1.11386068e-01, 9.68349889e-02,
    8.19379115e-02, 6.76403527e-02, 5.46630100e-02, 4.30616898e-02,
    3.39445445e-02, 2.62512556e-02, 2.02767989e-02, 1.58633226e-02,
    1.24495750e-02, 9.71666374e-03, 7.93086404e-03, 6.36669484e-03,
    5.32550983e-03, 4.42463766e-03, 3.77887973e-03, 3.22560071e-03,
    2.67072723e-03, 2.32632301e-03, 2.10469251e-03, 1.72521007e-03,
    1.49560725e-03, 1.21179265e-03, 9.85378764e-04, 8.83333067e-04,
    7.92448618e-04, 6.61702568e-04, 5.58062407e-04, 4.75150278e-04,
    3.84265829e-04, 3.49187620e-04, 2.67869955e-04, 2.42358531e-04,
    1.81768898e-04, 2.07280322e-04, 1.56257474e-04, 1.37123905e-04,
    1.16395874e-04, 1.16395874e-04, 7.01564169e-05, 7.33453449e-05,
    5.74007047e-05, 7.81287370e-05, 7.81287370e-05, 3.18892804e-05,
    3.18892804e-05, 1.91335682e-05, 3.50782084e-05, 2.23224963e-05,
    2.07280322e-05, 1.43501762e-05, 2.23224963e-05, 6.37785608e-06,
    1.27557122e-05, 1.43501762e-05, 6.37785608e-06, 7.97232011e-06,
    3.18892804e-06, 7.97232011e-06, 1.11612481e-05, 4.78339206e-06,
    3.18892804e-06, 3.18892804e-06, 3.18892804e-06, 3.18892804e-06
    ])
histogram = histogram/np.sum(histogram)
def neg_log_hist(dur, weight=None):
    score = -np.log(0 if dur >= len(histogram) else histogram[dur])
    if weight is None:
        return score
    else:
        return weight*(score) + np.log(np.sum(histogram**weight))


# Cache Gamma
# shape, loc, scale = (3, 0, 2.6)
shape, loc, scale = (3, 0, 2.5)
gamma_cache = []
for dur in range(200):
    gamma_cache.append(gamma.pdf(dur, shape, loc, scale))
def neg_log_gamma(dur, weight=None):
        # (
        # 2.967152765811849, -0.004979890790653328, 2.6549778308011014
        # )
    if dur < 200:
        score = -np.log(gamma_cache[dur])
    else:
        score = -np.log(gamma.pdf(dur, shape, loc, scale))
    if weight is None:
        return score
    else:
        return weight*score + np.log(np.sum(gamma_cache**weight))


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
    costs : n_frames*(n_frames + 1)/2 vector
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
        dur_weight=20**2, dur_weight_func=neg_chorowski, model_eos=False):

    # Hyperparameters
    # count_weight = 0
       
    # Distances between each z and each embedding (squared Euclidean)
    embedding_distances = distance.cdist(z, embedding, metric="sqeuclidean")
    # print("embedding_distances shape: {}".format(embedding_distances.shape))
    
    # Costs for segment intervals
    segment_intervals = get_segment_intervals(z.shape[0], n_max_frames)
    costs = np.inf*np.ones(len(segment_intervals))
    i_eos = segment_intervals[-1][-1]
    for i_seg, interval in enumerate(segment_intervals):
        if interval is None:
            continue
        i_start, i_end = interval
        dur = i_end - i_start
        if dur < n_min_frames:
            continue

        cost = np.min(
            np.sum(embedding_distances[i_start:i_end, :], axis=0)
            ) + dur_weight*dur_weight_func(dur)  # + count_weight

        # End-of-sequence
        if model_eos:
            alpha = 0.1
            K = 50
            if i_end == i_eos:
                cost += -np.log(alpha)
            else:
                cost += -np.log((1 - alpha)/K)

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
        dur_weight=0, n_frames_per_segment=7, n_min_segments=0,
        dur_weight_func=neg_chorowski):

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
        # cost = np.min(
        #     np.sum(embedding_distances[i_start:i_end, :], axis=0)
        #     ) - dur_weight*(dur - 1)
        cost = np.min(
            np.sum(embedding_distances[i_start:i_end, :], axis=0)
            ) + dur_weight*dur_weight_func(dur)
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


# Other promising options:
# - threshold="absolute", dependency="ftp"
# - threshold="absolute", dependency="mi"
def tp(utterance_list, threshold="relative", dependency="ftp"):
    from wordseg.algos import tp
    import wordseg.algos
    return list(
        tp.segment(utterance_list, threshold=threshold, dependency=dependency)
        )


def rasanen15(utterance_list, n_max=9, words_count_fn="words.tmp"):
    """
    The word decoding with n-grams approach of Räsänen et al. [Interspeech'15].

    See Section 2.3 in:

    - O, Räsänen, G. Doyle, M. C. Frank, "Unsupervised word discovery from
      speech using automatic segmentation into syllable-like units," in Proc.
      Interspeech, 2015.
    """
    from collections import Counter

    # Add space to beginning and end for matching
    tmp_list = []
    for utt in utterance_list:
        tmp_list.append(" " + utt + " ")
    utterance_list = tmp_list

    words = []
    counts = []
    for n in tqdm(range(n_max, 1, -1)):
        # Count n-grams
        n_gram_counter = Counter()
        for utt in utterance_list:
            utt = utt.split()
            n_grams = [tuple(utt[i:i + n]) for i in range(len(utt) - n + 1)]
            for n_gram in n_grams:
                n_gram_counter[n_gram] += 1

        # For all n-grams (of this order) occurring at least twice
        for n_gram in n_gram_counter:
            if n_gram_counter[n_gram] <= 1:
                continue
            # print(n_gram, n_gram_counter[n_gram])

            word = " " + "".join(n_gram) + " "
            word_unjoined = " " + " ".join(n_gram) + " "
            words.append(word)
            counts.append(n_gram_counter[n_gram])

            # # Temp
            # if word == " 445_83_102_456_ ":
            #     print (n_gram_counter[n_gram])
            #     for i_utt, utt in enumerate(utterance_list):
            #         if word_unjoined in utt:
            #             print(i_utt, utt)
            #     assert False

            # Replace occurrences
            for i_utt, utt in enumerate(utterance_list):
                if word_unjoined in utt:
                    utterance_list[i_utt] = utt.replace(word_unjoined, word)
                    # # Temp
                    # if i_utt in [228, 5569]:
                    #     print(word, word_unjoined)
                    #     print("!", utt)
                    #     print("!!", utterance_list[i_utt])


    if words_count_fn is not None:
        with open(words_count_fn, "w") as f:
            for word, count in zip(words, counts):
                f.write("{} {}\n".format(word, count))

    # Remove space at beginnning and end to match output format
    tmp_list = []
    for utt in utterance_list:
        tmp_list.append(utt.strip())
    utterance_list = tmp_list

    return utterance_list
