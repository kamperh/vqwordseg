#!/usr/bin/env python

"""
Converts a generated indices file to text.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from pathlib import Path
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
        "model", help="input VQ representations", choices=["vqvae", "vqcpc"]
        )
    parser.add_argument("dataset", type=str, help="input dataset")
    parser.add_argument(
        "split", type=str, help="input split", choices=["train", "val", "test"]
        )
    parser.add_argument("seg_tag", type=str, help="segmentation identifier")
    parser.add_argument("utt_key", type=str, help="utterance identifier")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    # Read indices
    indices_fn = (
        Path("exp")/args.model/args.dataset/args.split/args.seg_tag/
        "indices.npz"
        )
    print("Reading: {}".format(indices_fn))
    indices = np.load(indices_fn)

    if args.utt_key in indices:
        code_indices_intervals = indices[args.utt_key]
        indices_list = []
        for start, end, code in code_indices_intervals: 
            indices_list += [code]*(end - start) 
        index_fn = Path(args.utt_key).with_suffix(".txt")
        print("Writing: {}".format(index_fn))
        with open(index_fn, "w") as f:
            for code_index in indices_list:
                f.write("{}\n".format(code_index))
    else:
        print("Invalid key: {}".format(args.utt_key))
        print("Possible options include:")
        for key in list(indices)[:10]:
            print(key)



if __name__ == "__main__":
    main()
