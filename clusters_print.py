#!/usr/bin/env python

"""
Print the word clusters from large to small.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from collections import Counter
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import sys

from eval_segmentation import (
    get_intervals_from_dir, str_to_id_labels, intervals_to_max_overlap
    )
from utils import cluster_analysis


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
        choices=["vqvae", "vqcpc", "cpc_big"]
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


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Directories
    seg_dir = (
        Path("exp")/args.model/args.dataset/args.split/args.seg_tag/"intervals"
        )
    # phone_ref_dir = Path("data")/args.dataset/"phone_intervals"
    word_ref_dir = Path("data")/args.dataset/"word_intervals"

    # Read segmentation
    segmentation_interval_dict = {}
    print("Reading: {}".format(seg_dir))
    assert seg_dir.is_dir(), "missing directory: {}".format(seg_dir)
    segmentation_interval_dict = get_intervals_from_dir(seg_dir)
    utterances = segmentation_interval_dict.keys()

    # Read word reference
    print("Reading: {}".format(word_ref_dir))
    word_ref_interval_dict = get_intervals_from_dir(
        word_ref_dir, utterances
        )

    # Map e.g. "23_12_" to a unique integer ID e.g. 10
    if ("_" in list(segmentation_interval_dict.values())[0][0][-1]):
        segmentation_interval_dict, str_to_id, id_to_str = str_to_id_labels(
            segmentation_interval_dict
            )

    # Cluster analysis
    print()
    print("Analysing clusters")
    ref_interval_dict = word_ref_interval_dict
    pred_interval_dict = segmentation_interval_dict
    ref_labels = []
    pred_labels = []
    for utt in ref_interval_dict:
        ref = ref_interval_dict[utt]
        pred = pred_interval_dict[utt]
        ref_labels.extend(intervals_to_max_overlap(ref, pred))
        pred_labels.extend([int(i[2]) for i in pred])
    clusters = cluster_analysis.analyse_clusters(ref_labels, pred_labels)

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
    biggest_clusters = list(np.argsort(sizes)[-n_biggest:])  # http://stackoverflow.com/questions/16878715/how-to-find-the-index-of-n-largest-elements-in-a-list-or-np-array-python
    biggest_clusters.reverse()
    n_clusters_covered_90 = None
    print("-"*79)
    for i_cluster in biggest_clusters:

        # Raw cluster statistics
        cluster = clusters[i_cluster]
        lengths = lengths_dict[i_cluster]
        speakers = speakers_dict[i_cluster]

        print(
            f"Cluster {i_cluster} = '{id_to_str[i_cluster]}' "
            f"(rank: {i_cluster_count})"
            )

        print(f"Size: {cluster['size']}")
        print(f"Purity: {cluster['purity']*100:.2f}%")
        print(f"Mean length: {np.mean(lengths):.2f}")
        print(f"Std. length: {np.std(lengths):.2f}")

        print(f"Speakers: {Counter(speakers)}")

        # Tokens covered statistic
        n_tokens_covered += cluster["size"]
        prop_tokens_covered = n_tokens_covered*100./len(pred_labels)
        if n_clusters_covered_90 is None and prop_tokens_covered > 90.:
            n_clusters_covered_90 = i_cluster_count
        print(f"Tokens covered: {prop_tokens_covered:.2f}%")
        print("-"*79)

        i_cluster_count += 1


if __name__ == "__main__":
    main()

