#!/usr/bin/env python

"""
Generate a wav for a specified cluster.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2021
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import random
import subprocess
import sys
import uuid

from cluster_wav import cat_wavs, check_argv
from eval_segmentation import get_intervals_from_dir

audio_dir = Path(
    "/media/kamperh/endgame/datasets/zerospeech2020/2020/2017/"
    )
code_to_language = {
    "en": "english",
    "fr": "french",
    "zh": "mandarin"
    }

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    assert audio_dir.is_dir(), f"missing directory: {f}"

    # Language
    language_code = args.dataset[-2:]
    language = code_to_language[language_code]

    # Read segmentation
    seg_dir = (
        Path("exp")/args.model/args.dataset/args.split/args.seg_tag/"intervals"
        )
    segmentation_interval_dict = {}
    print("Reading: {}".format(seg_dir))
    assert seg_dir.is_dir(), "missing directory: {}".format(seg_dir)
    segmentation_interval_dict = get_intervals_from_dir(seg_dir)

    # Find matches
    tokens = []  # (utt_path, start, end),
                 # e.g. ("datasets/buckeye/s38/s3803a.wav", 413.97, 414.50)
    for utt_key in tqdm(segmentation_interval_dict):
        utt_key_split = utt_key.split("_")
        utt_start_end = utt_key_split[-1]
        utt_start, utt_end = utt_start_end.split("-")
        utt_start = int(utt_start)
        utt_end = int(utt_end)
        utt_label = "_".join(utt_key_split[:-1])
        utt_path = (
            audio_dir/language/args.split/utt_label
            ).with_suffix(".wav")
        for token_start, token_end, token_label in segmentation_interval_dict[
                utt_key]:
            if token_label == args.cluster_id:
                tokens.append((
                    utt_path, float(utt_start + token_start)/100.,
                    float(utt_start + token_end)/100.
                    ))

    # Create wav
    wav_fn = Path("{}.wav".format(args.cluster_id))
    cat_wavs(tokens, wav_fn, args.pad, args.shuffle)


if __name__ == "__main__":
    main()
