#!/usr/bin/env python

"""
Convert phone segmentation output to the ABX evaluation format.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2020
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import sys


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
    parser.add_argument("dataset", type=str, help="input dataset")
    parser.add_argument(
        "split", type=str, help="input split"
        )
    parser.add_argument("seg_tag", type=str, help="segmentation identifier")
    parser.add_argument(
        "--one_hot_k", type=int, default=None,
        help="store the ABX output as one-hot vectors instead of code vectors "
        "from the codebook; the maximum number of codes (exclusive) should be "
        "given"
        )
    # parser.add_argument(
    #     "model", type=str, help="the model type", choices=["vqvae", "vqcpc"]
    #     )
    # parser.add_argument("seg_tag", type=str, help="phone segmentation tag")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Read embeddings
    if args.one_hot_k is None:
        embedding_fn = Path("exp")/args.model/args.dataset/"embedding.npy"
        print("Reading:", embedding_fn)
        embeddings = np.load(embedding_fn)

    # Read indices
    indices_fn = (
        Path("exp")/args.model/args.dataset/args.split/args.seg_tag/
        "indices.npz"
        )
    print("Reading: {}".format(indices_fn))
    indices = np.load(indices_fn)

    # Make output directory
    abx_dir = Path("exp")/args.model/args.dataset/args.split/args.seg_tag/"abx"
    abx_dir.mkdir(exist_ok=True, parents=True)

    # Read indices and write codes
    print("Writing to: {}".format(abx_dir))
    for utt_key in tqdm(indices):
        indices_list = []

        for start, end, code in indices[utt_key]:
            indices_list += [code]*(end - start) 

        if args.one_hot_k is not None:
            one_hot_codes = np.zeros(
                (len(indices_list), args.one_hot_k), dtype=np.int32
                )
            one_hot_codes[np.arange(len(indices_list)), indices_list] = 1

            codes_fn = (abx_dir/utt_key).with_suffix(".txt")
            np.savetxt(codes_fn, one_hot_codes, fmt="%i")
        else:
            codes = embeddings[np.array(indices_list)]

            codes_fn = (abx_dir/utt_key).with_suffix(".txt")
            np.savetxt(codes_fn, codes, fmt="%.16f")

    # print("Now run:")
    # print("conda activate zerospeech2020")
    # print("export ZEROSPEECH2020_DATASET="
    #     "/media/kamperh/endgame/datasets/zerospeech2020/2020/"
    #     )
    # print("cd {}".format(base_dir))
    # print("zerospeech2020-evaluate 2019 -j4 abx/ -o abx_results.json")
    # print("cat abx_results.json")
    # print("cd -")

if __name__ == "__main__":
    main()
