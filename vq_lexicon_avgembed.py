#!/usr/bin/env python

"""
Build up a lexicon by clustering averaged HuBERT acoustic embeddings.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2023
"""

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import argparse
import faiss
import json
import math
import numpy as np
import sys

from eval_segmentation import get_intervals_from_dir


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
    )
    parser.add_argument(
        "model", help="input VQ representations"
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
    parser.add_argument(
        "--kmeans", type=int, 
        help="K-means is performed using this many clusters",
    )
    parser.add_argument(
        "--layer_tag", help="HuBERT layer to use", default="layer09"
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

    # Data

    # Directories
    seg_dir = (
        Path("exp")/args.model/args.dataset/args.split/args.seg_tag/"intervals"
    )

    # Read segmentation
    segmentation_interval_dict = {}
    print("Reading: {}".format(seg_dir))
    segmentation_interval_dict = get_intervals_from_dir(seg_dir)
    utterances = sorted(segmentation_interval_dict)

    # If Buckeye, check that utterances are correct
    if args.dataset == "buckeye":
        json_fn = (
            Path("../zerospeech2021_baseline/datasets")/args.dataset/args.split
        ).with_suffix(".json")
        utterances = []
        with open(json_fn) as f:
            metadata = json.load(f)
            for in_path, start, duration, out_path in metadata:
                utterances.append(str(Path(out_path).name))
    utterances = sorted(utterances)

    # Read features(prequantisation layer)
    features_dir = (
        Path(
            "../zerospeech2021_baseline/exp/buckeye/"
        )/args.model/args.split/args.layer_tag
    )
    features_dict = {}
    print("Reading from:", features_dir)
    for utt_key in tqdm(utterances):
        fn  = (features_dir/utt_key).with_suffix(".npy")
        features_dict[utt_key] = np.load(fn)

    # Feature intervals
    averaged_features = []
    for utt_key in tqdm(utterances):

        intervals = segmentation_interval_dict[utt_key]
        intervals = [(i[0], i[1]) for i in intervals]
        features = features_dict[utt_key]

        # Hack
        if intervals == [(0, 3)]:
            intervals = [(0, 4)]

        cur_averaged_features = []
        for start, end in intervals:
            start = math.ceil(start/2)
            end = math.ceil(end/2)
            awe = np.mean(features[start:end, :], axis=0)
            cur_averaged_features.append(awe)
       
        averaged_features.append(cur_averaged_features)


    # Clustering

    # Features for clustering
    X = np.vstack(averaged_features)
    print("X shape:", X.shape)
    print("Normalising features")
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X/norm

    # Cluster with FAISS
    print(datetime.now())
    print(f"Clustering: K = {args.kmeans}")
    D = X.shape[1]
    kmeans = faiss.Kmeans(
        D, args.kmeans, niter=10, nredo=5, verbose=True, gpu=True
        # D, args.kmeans, niter=1, nredo=1, verbose=True, gpu=True
    )
    kmeans.train(X)
    _, clusters = kmeans.index.search(X, 1)
    clusters = clusters.flatten()
    print(datetime.now())

    # Remap cluster labels to utterances
    i_embedding = 0
    output_seg_interval_dict = {}
    for i_utt, utt_key in tqdm(enumerate(utterances)):
        n_embeddings = len(averaged_features[i_utt])

        cur_segmentation_intervals = []
        for i_cur_embedding in range(n_embeddings):
            start, end, _ = segmentation_interval_dict[utt_key][i_cur_embedding]
            cur_segmentation_intervals.append(
                (start, end, clusters[i_embedding + i_cur_embedding])
            )
        output_seg_interval_dict[utt_key] = cur_segmentation_intervals

        i_embedding += n_embeddings


    # Output

    # Write intervals
    output_tag = f"wordseg_dpdp_avgembed_kmeans{args.kmeans}"
    output_dir = (
        Path("exp")/args.model/args.dataset/args.split/output_tag/"intervals"
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Writing to:", output_dir)
    for utt_key in tqdm(output_seg_interval_dict):
        with open((output_dir/utt_key).with_suffix(".txt"), "w") as f:
                for start, end, label in output_seg_interval_dict[utt_key]:
                    f.write("{:d} {:d} {}\n".format(start, end, label))


if __name__ == "__main__":
    main()
