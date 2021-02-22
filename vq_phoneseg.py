#!/usr/bin/env python

"""
Perform phone segmentation on VQ representations.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import sys

import algorithms


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
    parser.add_argument(
        "--input_format",
        help="format of input VQ representations (default: %(default)s)",
        choices=["npy", "txt"], default="txt"
        )
    parser.add_argument(
        "--algorithm",
        help="VQ segmentation algorithm (default: %(default)s)",
        choices=["dp_penalized", "dp_penalized_n_seg"], default="dp_penalized"
        )
    parser.add_argument(
        "--dur_weight", type=float,
        help="the duration penalty weight for the algorithm; if "
        "not specified, a sensible value is chosen based on the input model",
        default=None
        )
    parser.add_argument(
        "--output_tag", type=str, help="used to name the output directory; "
        "if not specified, the algorithm is used",
        default=None
        )
    parser.add_argument(
        "--downsample_factor", type=int,
        help="factor by which the VQ input is downsampled "
        "(default: %(default)s)",
        default=2
        )
    parser.add_argument(
        "--n_frames_per_segment", type=int,
        help="determines the number of segments for dp_penalized_n_seg "
        "(default: %(default)s)",
        default=7
        )
    parser.add_argument(
        "--n_min_segments", type=int,
        help="sets the minimum number of segments for dp_penalized_n_seg "
        "(default: %(default)s)", default=0
        )
    parser.add_argument(
        "--dur_weight_func",
        choices=["neg_log_geometric", "neg_log_poisson", "neg_log_hist",
        "neg_log_gamma"], default="neg_log_geometric",
        help="function to use for penalizing duration; "
        "if probabilistic, the negative log of the prior is used"
        )
    parser.add_argument(
        "--only_save_intervals", dest="only_save_intervals",
        action="store_true", help="if set, boundaries and indices are not "
        "saved as Numpy archives, only the interval text files are saved"
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

    # Command-line arguments
    segment_func = getattr(algorithms, args.algorithm)
    dur_weight_func = getattr(algorithms, args.dur_weight_func)
    if args.dur_weight is None:
        if args.model == "vqvae":
            args.dur_weight = 3
        elif args.model == "vqcpc":
            args.dur_weight = 20**2
        elif args.model == "cpc_big":
            args.dur_weight = 3
        else:
            assert False, "cannot set dur_weight automatically for model type"
        if args.algorithm == "dp_penalized_n_seg":
            args.dur_weight = 0
    if args.output_tag is None:
        args.output_tag = "phoneseg_{}".format(args.algorithm)

    # Directories and files
    input_dir = Path("exp")/args.model/args.dataset/args.split
    z_dir = input_dir/"auxiliary_embedding2"
    print("Reading: {}".format(z_dir))
    assert z_dir.is_dir(), "missing directory: {}".format(z_dir)
    if args.input_format == "npy":
        z_fn_list = sorted(list(z_dir.glob("*.npy")))
    elif args.input_format == "txt":
        z_fn_list = sorted(list(z_dir.glob("*.txt")))
    else:
        assert False, "invalid input format"

    # Read embedding matrix
    embedding_fn = input_dir.parent/"embedding.npy"
    print("Reading: {}".format(embedding_fn))
    embedding = np.load(embedding_fn)

    # Segment files one-by-one
    if not args.only_save_intervals:
        boundaries_dict[utt_key] = {}
        code_indices_dict[utt_key] = {}
    output_base_dir = input_dir/args.output_tag
    output_base_dir.mkdir(exist_ok=True, parents=True)
    print("Writing to: {}".format(output_base_dir))
    output_dir = output_base_dir/"intervals"
    output_dir.mkdir(exist_ok=True, parents=True)
    for input_fn in tqdm(z_fn_list):

        # Read pre-quantisation representations
        if args.input_format == "npy":
            z = np.load(input_fn)
        elif args.input_format == "txt":
            z = np.loadtxt(input_fn)

        # Segment
        if z.ndim == 1:
            continue
        if args.algorithm == "dp_penalized_n_seg":
            boundaries, code_indices = segment_func(
                embedding, z, dur_weight=args.dur_weight,
                n_frames_per_segment=args.n_frames_per_segment,
                n_min_segments=args.n_min_segments,
                dur_weight_func=dur_weight_func
                )
        else:
            boundaries, code_indices = segment_func(
                embedding, z, dur_weight=args.dur_weight,
                dur_weight_func=dur_weight_func
                )

        # Convert boundaries to same frequency as reference
        if args.downsample_factor > 1:
            boundaries_upsampled = np.zeros(
                len(boundaries)*args.downsample_factor, dtype=bool
                )
            for i, bound in enumerate(boundaries):
                boundaries_upsampled[i*args.downsample_factor + 1] = bound
            boundaries = boundaries_upsampled

            code_indices_upsampled = []
            for start, end, index in code_indices:
                code_indices_upsampled.append((
                    start*args.downsample_factor, 
                    end*args.downsample_factor,
                    index
                    ))
            code_indices = code_indices_upsampled

        if not args.only_save_intervals:
            boundaries_dict[utt_key] = boundaries
            code_indices_dict[utt_key] = code_indices

        # Write intervals
        utt_key = input_fn.stem
        with open((output_dir/utt_key).with_suffix(".txt"), "w") as f:
            for start, end, index in code_indices:
                f.write("{:d} {:d} {:d}\n".format(start, end, index))

    if not args.only_save_intervals:

        # Write code indices
        output_fn = output_base_dir/"indices.npz"
        print("Writing: {}".format(output_fn))
        np.savez_compressed(output_fn, **code_indices_dict)

        # Write boundaries
        output_fn = output_base_dir/"boundaries.npz"
        print("Writing: {}".format(output_fn))
        np.savez_compressed(output_fn, **boundaries_dict)


if __name__ == "__main__":
    main()
