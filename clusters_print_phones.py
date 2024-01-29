#!/usr/bin/env python

"""
Print the word clusters from large to small.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from collections import Counter
from pathlib import Path
from sklearn import metrics
from tqdm import tqdm
import argparse
import numpy as np

import sys

from eval_segmentation import get_intervals_from_dir, str_to_id_labels
from utils import cluster_analysis


prons = {
    "a": "ah",
    "ah": "aa",
    "all": "ao l",
    "and": "ae n",
    "he": "iy",
    "hey": "ey",
    "hm": "m",
    "huh": "hh ah",
    "i": "ay",
    "if": "ih f",
    "in": "ih n",
    "it": "ih t",
    "know": "n ow",
    "nah": "n eh",
    "no": "n ow",
    "of": "ah v",
    "oh": "ow",
    "okay": "ow k ey",
    "or": "ao r",
    "say": "s ey",
    "she": "sh iy",
    "so": "s ow",
    "that": "dh ae tq",
    "the": "dh ah",
    "then": "dh eh n",
    "there": "dh eh r",
    "they": "dh ey",
    "to": "t ih",
    "uh": "ah",
    "um": "ah m",
    "who": "hh uw",
    "wow": "w aw",
    "yeah": "y ae",
    "you": "y uw",
    "your": "y er",
    }


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "model", help="input VQ representations",
        choices=["vqvae", "vqcpc", "cpc_big", "eskmeans", "hubert"]
        )
    parser.add_argument("dataset", type=str, help="input dataset")
    parser.add_argument(
        "split", type=str, help="input split", choices=["train", "val", "test"]
        )
    parser.add_argument("seg_tag", type=str, help="segmentation identifier")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_sizes_purities(clusters):
    """Return two lists containing the sizes and purities of `clusters`."""
    sizes = []
    purities = []
    for cluster in clusters:
        sizes.append(cluster["size"])
        purities.append(cluster["purity"])
    return sizes, purities


def intervals_to_overlap(ref_intervals, pred_intervals, ref_labels=None):
    """
    Each interval is mapped to the reference label with maximum overlap.

    If `ref_labels` is not given, it is assumed that the labels are given as
    the third element of each of the elements in the `ref_intervals` list.
    """
    if ref_labels is None:
        ref_labels = [i[2] for i in ref_intervals]
    mapped_seq = []
    for pred_interval in pred_intervals:
        overlaps = []
        for ref_interval in ref_intervals:
            if ref_interval[1] <= pred_interval[0]:
                overlaps.append(0)
            elif ref_interval[0] >= pred_interval[1]:
                overlaps.append(0)
            else:
                overlap = pred_interval[1] - pred_interval[0]
                if ref_interval[0] > pred_interval[0]:
                    overlap -= (ref_interval[0] - pred_interval[0])
                if ref_interval[1] < pred_interval[1]:
                    overlap -= (pred_interval[1] - ref_interval[1])
                overlaps.append(overlap)
        mapped_seq.append(
            " ".join([j for i, j in enumerate(ref_labels) if overlaps[i] > 0])
            )
        # mapped_seq.append(ref_labels[np.argmax(overlaps)]) 
    return mapped_seq


def score_clusters(ref_interval_dict, pred_interval_dict, one_to_one=False):
    ref_labels = []
    pred_labels = []
    for utt in ref_interval_dict:
    # for utt in tqdm(ref_interval_dict):
        ref = ref_interval_dict[utt]
        pred = pred_interval_dict[utt]
        ref_labels.extend(intervals_to_overlap(ref, pred))
        pred_labels.extend([int(i[2]) for i in pred])

        # print(ref_labels)
        # print(pred_labels)
        # assert False

    pur = cluster_analysis.purity(ref_labels, pred_labels)

    if one_to_one:
        one_to_one, cluster_to_label_map = cluster_analysis.one_to_one_mapping(
            ref_labels, pred_labels
            )
    cluster_to_label_map_many = cluster_analysis.many_to_one_mapping(
        ref_labels, pred_labels
        )

    h, c, V = metrics.homogeneity_completeness_v_measure(
        ref_labels, pred_labels)
    
    if one_to_one:
        return pur, h, c, V, cluster_to_label_map, cluster_to_label_map_many
    else:
        return pur, h, c, V, cluster_to_label_map_many


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Directories
    seg_dir = (
        Path("exp")/args.model/args.dataset/args.split/args.seg_tag/"intervals"
        )
    phone_ref_dir = Path("data")/args.dataset/"phone_intervals"
    word_ref_dir = Path("data")/args.dataset/"word_intervals"

    # Read segmentation
    segmentation_interval_dict = {}
    print("Reading: {}".format(seg_dir))
    assert seg_dir.is_dir(), "missing directory: {}".format(seg_dir)
    segmentation_interval_dict = get_intervals_from_dir(seg_dir)
    utterances = segmentation_interval_dict.keys()

    # Read phone reference
    print("Reading: {}".format(phone_ref_dir))
    phone_ref_interval_dict = get_intervals_from_dir(
        phone_ref_dir, utterances
        )

    # Read word reference
    print("Reading: {}".format(word_ref_dir))
    word_ref_interval_dict = get_intervals_from_dir(
        word_ref_dir, utterances
        )

    # Fix missing phone references
    for utt in word_ref_interval_dict:
        if utt not in phone_ref_interval_dict:
            phone_ref_interval_dict[utt] = []
            for start, end, label in word_ref_interval_dict[utt]:
                if label in prons:
                    label = prons[label]
                phone_ref_interval_dict[utt].append((start, end, label))

    # Map e.g. "23_12_" to a unique integer ID e.g. 10
    # if ("_" in list(segmentation_interval_dict.values())[0][0][-1]):
    if True:
        segmentation_interval_dict, str_to_id, id_to_str = str_to_id_labels(
            segmentation_interval_dict
            )

    # Cluster analysis
    print()
    print("Analysing clusters")
    # ref_interval_dict = word_ref_interval_dict
    ref_interval_dict = phone_ref_interval_dict
    pred_interval_dict = segmentation_interval_dict
    ref_labels = []
    pred_labels = []
    for utt in ref_interval_dict:
        ref = ref_interval_dict[utt]
        pred = pred_interval_dict[utt]
        ref_labels.extend(intervals_to_overlap(ref, pred))
        pred_labels.extend([int(i[2]) for i in pred])

    clusters = cluster_analysis.analyse_clusters(ref_labels, pred_labels)
    purity, h, c, V, cluster_to_label_map_many = score_clusters(
        phone_ref_interval_dict, segmentation_interval_dict
        )

    # Additional cluster analysis
    gender_purity = 0
    speaker_purity = 0
    lengths_dict = {}   # lengths_dict[105] are lengths of cluster 105
    speakers_dict = {}  # speakers_dict[105] are speakers of cluster 105
    for utt_key in segmentation_interval_dict:
        speaker = utt_key.split("_")[0]
        for start, end, label in segmentation_interval_dict[utt_key]:
            if label not in lengths_dict:
                lengths_dict[label] = []
                speakers_dict[label] = []
            lengths_dict[label].append(end - start)
            speakers_dict[label].append(speaker)

    # Print the biggest clusters
    n_biggest = 20
    n_tokens_covered = 0
    i_cluster_count = 1
    sizes, _ = get_sizes_purities(clusters)
    biggest_clusters = list(np.argsort(sizes)[-n_biggest:])
    biggest_clusters.reverse()
    n_clusters_covered_90 = None
    print("-"*79)
    overlap_dict = {}
    show_clusters_list = []
    show_clusters_sizes = []    
    for i_cluster in biggest_clusters:

        # Raw cluster statistics
        cluster = clusters[i_cluster]
        counts = cluster["counts"]
        sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=True)
        # assert False
        lengths = lengths_dict[i_cluster]
        speakers = speakers_dict[i_cluster]

        print(
            f"Cluster {i_cluster} = '{id_to_str[i_cluster]}' "
            f"(rank: {i_cluster_count})"
            )
        if i_cluster in cluster_to_label_map_many:
            print(f"Mapped to: '{cluster_to_label_map_many[i_cluster]}'")

        print(f"Size: {cluster['size']}")
        print(f"Purity: {cluster['purity']*100:.2f}%")
        print(f"Mean length: {np.mean(lengths):.2f}")
        print(f"Std. length: {np.std(lengths):.2f}")

        print(f"Speakers: {Counter(speakers)}")
        print("Counts:", sorted_counts)

        # Tokens covered statistic
        n_tokens_covered += cluster["size"]
        prop_tokens_covered = n_tokens_covered*100./len(pred_labels)
        if n_clusters_covered_90 is None and prop_tokens_covered > 90.:
            n_clusters_covered_90 = i_cluster_count
        print(f"Tokens covered: {prop_tokens_covered:.2f}%")
        print("-"*79)

        i_cluster_count += 1

        # print(counts.items())
        # assert False

        phone_counter = Counter()
        for i, c in counts.items():
            for phone in i.split():
                phone_counter[phone] += c

        sorted_counts = sorted(phone_counter.items(), key=lambda x:x[1], reverse=True)
        sorted_counts = [(i[0], i[1]) for i in sorted_counts if i[1] > 0]  # remove counts that occur once
        overlap_dict[i_cluster] = sorted_counts
        show_clusters_sizes.append(sum([i[1] for i in sorted_counts]))

        show_clusters_list.append(i_cluster)        

    import matplotlib.pyplot as plt
    cmap = "Blues"

    # Sort according to shown sizes
    show_clusters_list = [x for (y, x) in sorted(zip(show_clusters_sizes, show_clusters_list))]
    show_clusters_list.reverse()

    mapping_array = np.zeros((
        len(overlap_dict), max([len(overlap_dict[i]) for i in sorted(overlap_dict)])
        ))
    for i, i_cluster in enumerate(show_clusters_list):
        overlaps = [j[1] for j in overlap_dict[i_cluster]]
        mapping_array[i, :len(overlaps)] = overlaps

    fig, ax = plt.subplots()
    heatmap = ax.imshow(mapping_array, cmap=cmap, interpolation="nearest", aspect="auto")
    plt.yticks(range(len(show_clusters_list)), show_clusters_list)
    plt.xticks([])
    for y, i_cluster in enumerate(show_clusters_list):
        for x, (phone, count) in enumerate(overlap_dict[i_cluster]):
            plt.text(x, y, phone, horizontalalignment="center", verticalalignment="center")
    plt.colorbar(heatmap)
    plt.xlim([-0.5, 5])

    plt.show()

if __name__ == "__main__":
    main()

