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
        "model", help="input VQ representations", choices=["vqvae", "vqcpc"]
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
        if args.algorithm == "dp_penalized_n_seg":
            args.dur_weight = 0
    if args.output_tag is None:
        args.output_tag = "phoneseg_{}".format(args.algorithm)

    # Read pre-quantisation representations
    input_dir = Path("exp")/args.model/args.dataset/args.split
    z_dir = input_dir/"auxiliary_embedding2"
    print("Reading: {}".format(z_dir))
    assert z_dir.is_dir(), "missing directory: {}".format(z_dir)
    z_dict = {}
    if args.input_format == "npy":
        for input_fn in tqdm(z_dir.glob("*.npy")):
            z_dict[input_fn.stem] = np.load(input_fn)
    elif args.input_format == "txt":
        for input_fn in tqdm(z_dir.glob("*.txt")):
            z_dict[input_fn.stem] = np.loadtxt(input_fn)
    else:
        assert False, "invalid input format"

    # Read embedding matrix
    embedding_fn = input_dir.parent/"embedding.npy"
    print("Reading: {}".format(embedding_fn))
    embedding = np.load(embedding_fn)

    # Segmentation
    boundaries_dict = {}
    code_indices_dict = {}
    print("Running {}:".format(args.algorithm))
    for utt_key in tqdm(z_dict):

        # Segment
        z = z_dict[utt_key]
        if z.ndim == 1:
            continue
        if args.algorithm == "dp_penalized_n_seg":
            boundaries, code_indices = segment_func(
                embedding, z, dur_weight=args.dur_weight,
                n_frames_per_segment=args.n_frames_per_segment,
                n_min_segments=args.n_min_segments,
                dur_weight_func=dur_weight_func
                )
            # print(args.dur_weight,
            #     args.n_frames_per_segment,
            #     args.n_min_segments
            #     )
            # assert False
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

        boundaries_dict[utt_key] = boundaries_upsampled
        code_indices_dict[utt_key] = code_indices

    output_base_dir = input_dir/args.output_tag
    output_base_dir.mkdir(exist_ok=True, parents=True)
    print("Writing to: {}".format(output_base_dir))

    # Write code indices
    output_fn = output_base_dir/"indices.npz"
    print("Writing: {}".format(output_fn))
    np.savez_compressed(output_fn, **code_indices_dict)
    # output_dir = output_base_dir/"indices"
    # output_dir.mkdir(exist_ok=True, parents=True)
    # # print("Writing to: {}".format(output_dir))
    # for utt_key in tqdm(code_indices_dict):
    #     np.save(
    #         (output_dir/utt_key).with_suffix(".npy"),
    #         np.array([i[-1] for i in code_indices_dict[utt_key]],
    #         dtype=np.int)
    #         )

    # Write boundaries
    output_fn = output_base_dir/"boundaries.npz"
    print("Writing: {}".format(output_fn))
    np.savez_compressed(output_fn, **boundaries_dict)
    # output_dir = output_base_dir/"boundaries"
    # output_dir.mkdir(exist_ok=True, parents=True)
    # # print("Writing to: {}".format(output_dir))
    # for utt_key in tqdm(code_indices_dict):
    #     np.save(
    #         (output_dir/utt_key).with_suffix(".npy"),
    #         np.array(boundaries_dict[utt_key], dtype=np.bool)
    #         )

    # Write intervals
    output_dir = output_base_dir/"intervals"
    output_dir.mkdir(exist_ok=True, parents=True)
    # print("Writing to: {}".format(output_dir))
    for utt_key in tqdm(code_indices_dict):
        with open((output_dir/utt_key).with_suffix(".txt"), "w") as f:
            for start, end, index in code_indices_dict[utt_key]:
                f.write("{:d} {:d} {:d}\n".format(start, end, index))


if __name__ == "__main__":
    main()
