#!/usr/bin/env python

"""
Convert intervals to ZeroSpeech format.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import sys

from eval_segmentation import get_intervals_from_dir, str_to_id_labels


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
        choices=["vqvae", "vqcpc", "cpc_big", "xlsr", "hubert"]
        )
    parser.add_argument(
        "dataset", type=str, help="input dataset"
        )
    parser.add_argument(
        "split", type=str, help="input split", choices=["train", "val", "test"]
        )
    parser.add_argument(
        "seg_tag", type=str, help="segmentation identifier"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Read segmentation
    seg_dir = (
        Path("exp")/args.model/args.dataset/args.split/args.seg_tag/"intervals"
        )
    segmentation_interval_dict = {}
    print("Reading: {}".format(seg_dir))
    assert seg_dir.is_dir(), "missing directory: {}".format(seg_dir)
    segmentation_interval_dict = get_intervals_from_dir(seg_dir)

    # print(segmentation_interval_dict["A36_018467-018616"])

    # Map e.g. "23_12_" to a unique integer ID e.g. 10
    segmentation_interval_dict, str_to_id, id_to_str = str_to_id_labels(
        segmentation_interval_dict
        )

    # Get items in each cluster
    clusters = {}
    for i_cluster in range(len(id_to_str)):
        clusters[i_cluster] = []
    for utt_key in tqdm(segmentation_interval_dict):
        utt_key_split = utt_key.split("_")
        utt_start_end = utt_key_split[-1]
        utt_start, utt_end = utt_start_end.split("-")
        # utt_label, interval = utt_key.split("_")
        # utt_start, utt_end = interval.split("-")
        utt_start = int(utt_start)
        utt_end = int(utt_end)
        utt_label = "_".join(utt_key_split[:-1])
        for token_start, token_end, cluster in (
                segmentation_interval_dict[utt_key]):
            clusters[cluster].append((
                utt_label,
                float(utt_start + token_start)/100.,
                float(utt_start + token_end)/100.
                ))

    # Write clusters
    clusters_fn = seg_dir.parent/"clusters.txt"
    print(f"Writing: {clusters_fn}")
    n_tokens = 0
    n_classes = 0    
    with open(clusters_fn, "w") as f:
        for c in sorted(clusters):
            n_classes += 1
            # print(c)
            f.write(f"Class {c}\n")
            for utt, start, end in sorted(clusters[c]):
                f.write(f"{utt} {start:.4f} {end:.4f}\n")
                n_tokens += 1
            f.write("\n")

    # Write clusters to IDs
    clusters_to_ids_fn = seg_dir.parent/"clusters_to_ids.txt"
    print(f"Writing: {clusters_to_ids_fn}")
    with open(clusters_to_ids_fn, "w") as f:
        for i_cluster in range(len(id_to_str)):
            f.write(f"{i_cluster} {id_to_str[i_cluster]}\n")
    print(f"No. of classes: {n_classes}")
    print(f"No. of tokens: {n_tokens}")


if __name__ == "__main__":
    main()
