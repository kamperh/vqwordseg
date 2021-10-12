#!/usr/bin/env python

"""
Perform word segmentation on VQ representations.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import sys

from utils import eval_segmentation
from vqwordseg import algorithms


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
    parser.add_argument(
        "phoneseg_tag", type=str, help="input phone segmentation"
        )
    parser.add_argument(
        "--algorithm",
        help="word segmentation algorithm (default: %(default)s)",
        choices=["ag", "tp", "rasanen15"], default="ag"
        )
    parser.add_argument(
        "--output_tag", type=str, help="used to name the output directory; "
        "if not specified, the algorithm is used",
        default=None
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
    if args.output_tag is None:
        args.output_tag = "wordseg_{}_{}".format(
            args.algorithm,
            args.phoneseg_tag.replace("phoneseg_", "")
            )

    # Phone intervals
    input_dir = (
        Path("exp")/args.model/args.dataset/args.split/
        args.phoneseg_tag/"intervals"
        )
    phoneseg_interval_dict = {}
    print("Reading: {}".format(input_dir))
    assert input_dir.is_dir(), "missing directory: {}".format(input_dir)
    phoneseg_interval_dict = eval_segmentation.get_intervals_from_dir(
        input_dir
        )
    utterances = phoneseg_interval_dict.keys()

    # # Temp
    # print(list(utterances)[228], list(utterances)[5569])
    # assert False

    # Segmentation
    print("Segmenting:")
    prepared_text = []
    for utt_key in utterances:
        prepared_text.append(
            " ".join([i[2] + "_" for i in phoneseg_interval_dict[utt_key]])
            )
    word_segmentation = segment_func(prepared_text)
    # print(prepared_text[:10])
    wordseg_interval_dict = {}
    for i_utt, utt_key in tqdm(enumerate(utterances)):
        words_segmented = word_segmentation[i_utt].split(" ")
        word_start = 0
        word_label = ""
        i_word = 0
        wordseg_interval_dict[utt_key] = []
        for (phone_start,
                phone_end, phone_label) in phoneseg_interval_dict[utt_key]:
            word_label += phone_label + "_"
            if words_segmented[i_word] == word_label:
                wordseg_interval_dict[utt_key].append((
                    word_start, phone_end, word_label
                    ))
                word_label = ""
                word_start = phone_end
                i_word += 1

    # Write intervals
    output_dir = (
        Path("exp")/args.model/args.dataset/args.split/
        args.output_tag/"intervals"
        )
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Writing to: {}".format(output_dir))
    for utt_key in tqdm(wordseg_interval_dict):
        with open((output_dir/utt_key).with_suffix(".txt"), "w") as f:
            for start, end, label in wordseg_interval_dict[utt_key]:
                f.write("{:d} {:d} {}\n".format(start, end, label))


if __name__ == "__main__":
    main()
