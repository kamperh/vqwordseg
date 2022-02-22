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

from eval_segmentation import get_intervals_from_dir

buckeye_audio_dir = Path("/home/kamperh/endgame/datasets/buckeye")


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
    parser.add_argument(
        "cluster_id", type=str,
        help="the code or cluster, e.g. '20' or '158_111_'"
        )
    parser.add_argument(
        "--pad", type=float, default=0.25,
        help="if given, add padding between tokens (default: %(default)s)"
        )
    parser.add_argument(
        "--no_shuffle", dest="shuffle", action="store_false",
        help="do not shuffle tokens, sort them by utterance label"
        )
    parser.set_defaults(shuffle=True)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def cat_wavs(tokens, wav_fn, pad=None, shuffle=True):

    if wav_fn.is_file():
        print("Warning: Deleting {}".format(wav_fn))
        wav_fn.unlink()

    tmp_basename = str(uuid.uuid4())
    tmp_wav = Path(tmp_basename).with_suffix(".wav")

    if shuffle:
        random.seed(1)
        random.shuffle(tokens)
    else:
        tokens = sorted(tokens)

    print("Writing: {}".format(wav_fn))
    for utt_path, start, end in tqdm(tokens):
        duration = end - start
        sox_cmd = [
            "sox", str(utt_path), str(tmp_wav), "trim", str(start),
            str(duration)
            ]
        if pad is not None:
            sox_cmd += ["pad", "0", str(pad)]

        # Cut out using sox
        result = subprocess.run(sox_cmd)
        assert result.returncode == 0

        # Concatenate wavs
        if wav_fn.is_file():
            tmp_wav2 = Path(tmp_basename).with_suffix(".2.wav")
            sox_cmd = ["sox", str(wav_fn), str(tmp_wav), str(tmp_wav2)]
            result = subprocess.run(sox_cmd)
            assert result.returncode == 0
            tmp_wav2.rename(wav_fn)
            tmp_wav.unlink()
        else:
            tmp_wav.rename(wav_fn)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    assert buckeye_audio_dir.is_dir(), "missing directory: {}".format(
        buckeye_audio_dir
        )

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
        # print(utt_key)
        speaker, utt, utt_start_end = utt_key.split("_")
        utt_start, utt_end = utt_start_end.split("-")
        utt_start = int(utt_start)
        utt_end = int(utt_end)
        utt_label = speaker + utt
        utt_path = (buckeye_audio_dir/speaker/utt_label).with_suffix(".wav")
        for token_start, token_end, token_label in segmentation_interval_dict[
                utt_key]:
            if token_label == args.cluster_id:
                tokens.append((
                    utt_path, float(utt_start + token_start)/100.,
                    float(utt_start + token_end)/100.
                    ))

    # # Temp
    # print(tokens)

    # Create wav
    wav_fn = Path("{}.wav".format(args.cluster_id))
    cat_wavs(tokens, wav_fn, args.pad, args.shuffle)


if __name__ == "__main__":
    main()
