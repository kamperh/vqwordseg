"""
VQ phone and word segmentation algorithms.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021, 2023
"""

from datetime import datetime
from pathlib import Path
from scipy.spatial import distance
from scipy.special import factorial
from scipy.stats import gamma
from tqdm import tqdm
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent/"../../dpdp_aernn"))


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


def dp_penalized_hsmm(embedding, z, n_min_frames=0, n_max_frames=15,
        dur_weight=20**2, dur_weight_func=neg_log_gamma, model_eos=False):
    """Segmentation using a hidden semi-Markov model (HSMM)."""

    # Hyperparameters
    # count_weight = 0
    sigma = 1.0/dur_weight
    D = z.shape[1]
       
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

        cost = (
            1/(2*sigma**2)*np.min(
            np.sum(embedding_distances[i_start:i_end, :], axis=0)
            )
            + 0.5*dur*D*np.log(2*np.pi) + 0.5*dur*D*np.log(sigma**2)
            + dur_weight_func(dur)  # + count_weight
            )

        # End-of-sequence
        if model_eos:
            alpha = 0.1  # 0.1
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
    
    n_max_symbols = 50  # 100
    for i_utt in range(len(utterance_list)):
        utterance = utterance_list[i_utt]
        utterance_list[i_utt] = (
            "_ ".join(utterance[:-1].split("_ ")[:n_max_symbols]) + "_"
            )

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
                f.write(f"{word} {count}\n")

    # Remove space at beginnning and end to match output format
    tmp_list = []
    for utt in utterance_list:
        tmp_list.append(utt.strip())
    utterance_list = tmp_list

    return utterance_list


def dpdp_aernn(utterance_list, dur_weight=3.0, kmeans=None):
    """
    Uses an AE-RNN as the scoring function for DPDP.

    Parameters
    ----------
    kmeans: int
        If provided, K-means is performed on the resulting AE-RNN embeddings
        using this many clusters.
    """

    from dpdp_aernn import datasets, models, viterbi
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn

    # Random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # DATA

    # Convert to format for DPDP AE-RNN
    prepared_text = []
    for sentence in utterance_list:
        prepared_text.append(sentence.replace(" ", "").strip("_"))

    # Vocabulary
    PAD_SYMBOL      = "<pad>"
    SOS_SYMBOL      = "<s>"    # start of sentence
    EOS_SYMBOL      = "</s>"   # end of sentence
    BOUNDARY_SYMBOL = " "      # word boundary
    symbols = set()
    for sentence in prepared_text:
        for char in sentence.split("_"):
            symbols.add(char)
    SYMBOLS = [PAD_SYMBOL, SOS_SYMBOL, EOS_SYMBOL, BOUNDARY_SYMBOL] + (sorted(list(symbols)))
    symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}
    id_to_symbol = {i: s for i, s in enumerate(SYMBOLS)}

    def text_to_id(text, add_sos_eos=False):
        """
        Convert text to a list of symbol IDs.

        Sentence start and end symbols can be added by setting `add_sos_eos`.
        """
        symbol_ids = []
        for word in text.split(" "):
            for code in word.split("_"):
                if code == "":
                    continue
                symbol_ids.append(symbol_to_id[code])
            symbol_ids.append(symbol_to_id[BOUNDARY_SYMBOL])
        symbol_ids = symbol_ids[:-1]  # remove last space

        if add_sos_eos:
            return [
                symbol_to_id[SOS_SYMBOL]] + symbol_ids + [symbol_to_id[EOS_SYMBOL]
                ]
        else:
            return symbol_ids

    # print(text_to_id(prepared_text[0]))
    # print([id_to_symbol[i] for i in text_to_id(prepared_text[0])])
    cur_train_sentences = prepared_text
    # cur_train_sentences = prepared_text[:10000]
    # print("Warning: Only training on first 10k sentences")


    # MODEL

    # AE-RNN model
    n_symbols = len(SYMBOLS)
    symbol_embedding_dim = 10  # 25
    hidden_dim = 500  # 250  # 500  # 1000  # 200
    embedding_dim = 50  # 150  # 300  # 25
    teacher_forcing_ratio = 0.5  # 1.0  # 0.5  # 1.0
    n_encoder_layers = 1  # 1  # 3  # 10
    n_decoder_layers = 1  # 1  # 1
    batch_size = 32  # 32*3  # 32
    learning_rate = 0.001
    input_dropout = 0.0  # 0.0 # 0.5
    dropout = 0.0
    n_symbols_max = 50  # 25  # 50
    # n_epochs_max = 5
    n_epochs_max = None     # determined from n_max_steps and batch size
    n_steps_max = 1500      # 2500  # 1500  # 1000  # None
    # n_steps_max = None    # only use n_epochs_max
    bidirectional_encoder = False

    encoder = models.Encoder(
        n_symbols=n_symbols,
        symbol_embedding_dim=symbol_embedding_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_layers=n_encoder_layers,
        dropout=dropout,
        input_dropout=input_dropout,
        bidirectional=bidirectional_encoder
        )
    # decoder = models.Decoder1(
    #     n_symbols=n_symbols,
    #     symbol_embedding_dim=symbol_embedding_dim,
    #     hidden_dim=hidden_dim,
    #     embedding_dim=embedding_dim,
    #     n_layers=n_decoder_layers,
    #     sos_id = symbol_to_id[SOS_SYMBOL],
    #     teacher_forcing_ratio=teacher_forcing_ratio,
    #     dropout=dropout
    #     )
    decoder = models.Decoder2(
        n_symbols=n_symbols,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_layers=n_decoder_layers,
        dropout=dropout
        )
    model = models.EncoderDecoder(encoder, decoder)    


    # TRAINING

    # Training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training data
    train_dataset = datasets.WordDataset(
        cur_train_sentences, text_to_id, n_symbols_max=n_symbols_max
        )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=datasets.pad_collate
        )

    # # Validation data    # val_dataset = datasets.WordDataset(cur_val_sentences, text_to_id)
    # val_loader = DataLoader(
    #     val_dataset, batch_size=batch_size, shuffle=True,
    #     collate_fn=datasets.pad_collate
    #     )

    # Loss
    criterion = nn.NLLLoss(
        reduction="sum", ignore_index=symbol_to_id[PAD_SYMBOL]
        )
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if n_epochs_max is None:
        steps_per_epoch = np.ceil(len(cur_train_sentences)/batch_size)
        n_epochs_max = int(np.ceil(n_steps_max/steps_per_epoch))

    print("Training AE-RNN:")
    i_step = 0
    for i_epoch in range(n_epochs_max):

        # Training
        model.train()
        train_losses = []
        for i_batch, (data, data_lengths) in enumerate(tqdm(train_loader)):
            optimiser.zero_grad()
            data = data.to(device)       
            encoder_embedding, decoder_output = model(
                data, data_lengths, data, data_lengths
                )

            loss = criterion(
                decoder_output.contiguous().view(-1, decoder_output.size(-1)),
                data.contiguous().view(-1)
                )
            loss /= len(data_lengths)
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())
            i_step += 1
            if i_step == n_steps_max and n_steps_max is not None:
                break
        
        # # Validation
        # model.eval()
        # val_losses = []
        # with torch.no_grad():
        #     for i_batch, (data, data_lengths) in enumerate(val_loader):
        #         data = data.to(device)            
        #         encoder_embedding, decoder_output = model(
        #             data, data_lengths, data, data_lengths
        #             )

        #         loss = criterion(
        #             decoder_output.contiguous().view(-1,
        #             decoder_output.size(-1)), data.contiguous().view(-1)
        #             )
        #         loss /= len(data_lengths)
        #         val_losses.append(loss.item())
        
        # print(
        #     "Epoch {}, train loss: {:.3f}, val loss: {:.3f}".format(
        #     i_epoch,
        #     np.mean(train_losses),
        #     np.mean(val_losses))
        #     )
        print(
            f"Epoch {i_epoch}, train loss: {np.mean(train_losses):.3f}"
            )
        sys.stdout.flush()

        if i_step == n_steps_max and n_steps_max is not None:
            break


    # SEGMENTATION

    def get_segmented_sentence(ids, boundaries):
        output = ""
        cur_word = []
        for i_symbol, boundary in enumerate(boundaries):
            cur_word.append(id_to_symbol[ids[i_symbol]])
            if boundary:
                output += "_".join(cur_word) + "_"
                output += " "
                cur_word = []
        return output.strip()

    # Embed segments
    eval_sentences = prepared_text

    # Random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Data
    sentences = eval_sentences
    # sentences = cur_val_sentences
    interval_dataset = datasets.SentenceIntervalDataset(
        sentences,
        text_to_id,
        join_char="_"
        )
    segment_loader = DataLoader(
        interval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=datasets.pad_collate,
        drop_last=False
        )

    # Apply model to data
    model.decoder.teacher_forcing_ratio = 1.0
    model.eval()
    rnn_losses = []
    lengths = []
    print("Embedding segments:")
    with torch.no_grad():
        for i_batch, (data, data_lengths) in enumerate(tqdm(segment_loader)):
            data = data.to(device)
            
            encoder_embedding, decoder_output = model(
                data, data_lengths, data, data_lengths
                )

            for i_item in range(data.shape[0]):
                item_loss = criterion(
                    decoder_output[i_item].contiguous().view(-1,
                    decoder_output[i_item].size(-1)),
                    data[i_item].contiguous().view(-1)
                    )
                rnn_losses.append(item_loss.cpu().numpy())
                lengths.append(data_lengths[i_item])

    # Segment
    i_item = 0
    losses = []
    cur_segmented_sentences = []
    print("Segmenting:")
    for i_sentence, intervals in enumerate(tqdm(interval_dataset.intervals)):
        
        # Costs for segment intervals
        costs = np.inf*np.ones(len(intervals))
        i_eos = intervals[-1][-1]
        for i_seg, interval in enumerate(intervals):
            if interval is None:
                continue
            i_start, i_end = interval
            dur = i_end - i_start
            assert dur == lengths[i_item]
            eos = (i_end == i_eos)  # end-of-sequence
            
            # Chorowski
            costs[i_seg] = (
                rnn_losses[i_item]
                + dur_weight*neg_chorowski(dur)
                )
            
    #         # Gamma
    #         costs[i_seg] = (
    #             rnn_losses[i_item]
    #             + dur_weight*neg_log_gamma(dur)
    #             + np.log(np.sum(gamma_cache**dur_weight))
    #             )
            
    #         # Poisson
    #         costs[i_seg] = (
    #             rnn_losses[i_item]
    #             + neg_log_poisson(dur)
    #             )

    #         # Histogram
    #         costs[i_seg] = (
    #             rnn_losses[i_item]
    #             + dur_weight*(neg_log_hist(dur))
    #             + np.log(np.sum(histogram**dur_weight))
    #             )
        
    #         # Sequence boundary
    #         alpha = 0.3  # 0.3  # 0.9
    #         if eos:
    #             costs[i_seg] += -np.log(alpha)
    #         else:
    #             costs[i_seg] += -np.log(1 - alpha)
    # #             K = 5000
    # #             costs[i_seg] += -np.log((1 - alpha)/K)

            # Temp
    #         if dur > 10 or dur <= 1:
    #             costs[i_seg] = +np.inf
            i_item += 1
        
        # Viterbi segmentation
        n_frames = len(interval_dataset.sentences[i_sentence])
        summed_cost, boundaries = viterbi.custom_viterbi(costs, n_frames)
        losses.append(summed_cost)
        
        reference_sentence = sentences[i_sentence]
        segmented_sentence = get_segmented_sentence(
                interval_dataset.sentences[i_sentence],
                boundaries
                )
        cur_segmented_sentences.append(segmented_sentence)
    #     # Print examples of the first few sentences
    #     if i_sentence < 10:
    #         print(reference_sentence)
    #         print(segmented_sentence)
    #         print()
        
    print(f"NLL: {np.sum(losses):.4f}")

    # # Temp
    print(cur_segmented_sentences[0])
    # # assert False

    if kmeans is None:
        return cur_segmented_sentences


    # K-MEANS CLUSTERING

    print("K-means clustering:")
    clustering_sentences = cur_segmented_sentences
    K = kmeans

    # Data
    train_dataset = datasets.WordDataset(
        clustering_sentences, text_to_id, n_symbols_max=n_symbols_max
        )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=datasets.pad_collate,
        drop_last=False
        )

    # Apply model to data
    print("Embedding segments:")
    model.eval()
    encoder_embeddings = []
    with torch.no_grad():
        for i_batch, (data, data_lengths) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            encoder_embedding, decoder_output = model(
                data, data_lengths, data, data_lengths
                )
            encoder_embeddings.append(encoder_embedding.cpu().numpy())
    X = np.vstack(encoder_embeddings)
    print("X shape:", X.shape)

    print("Normalizing embeddings")
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X/norm

    # Cluster (scikit-learn)
    """
    from sklearn import cluster
    print(datetime.now())
    print(f"Clustering: K = {K}")
    kmeans_model = cluster.KMeans(n_clusters=K, max_iter=10)
    kmeans_model.fit(X)
    print("Inertia: {:.4f}".format(kmeans_model.inertia_))
    clusters = kmeans_model.predict(X)
    print(datetime.now())
    """

    # Cluster (FAISS)
    import faiss
    print(datetime.now())
    print(f"Clustering: K = {K}")
    D = X.shape[1]
    kmeans = faiss.Kmeans(D, K, niter=20, nredo=20, verbose=True, gpu=True)
    kmeans.train(X)
    _, clusters = kmeans.index.search(X, 1)
    clusters = clusters.flatten()
    print(datetime.now())

    # Cluster labels for current segmentation
    i_embedding = 0
    clustered_sentences = []
    for i_utt in tqdm(range(len(clustering_sentences))):
        n_embeddings = len(clustering_sentences[i_utt].split(" "))
        cur_clusters = []
        for i_cur_embedding in range(n_embeddings):
            cur_clusters.append(clusters[i_embedding + i_cur_embedding])
        clustered_sentences.append(cur_clusters)
        i_embedding += n_embeddings

    # print(clustered_sentences[0])
    # print(len(clustering_sentences))
    # print(len(clustered_sentences))

    # clustered_sentences_str = []
    # for i_utt in tqdm(range(len(clustered_sentences))):
    #     clustered_sentences_str.append(
    #         " ".join([f"{i}_" for i in clustered_sentences[i_utt]])
    #         )

    return cur_segmented_sentences, clustered_sentences
