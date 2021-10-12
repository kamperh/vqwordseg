#!/usr/bin/env python

"""
Evaluate segmentation output.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2020
"""

from pathlib import Path
from sklearn import metrics
from tqdm import tqdm
import argparse
import numpy as np
import sys

# from . import cluster_analysis
from cluster_analysis import purity


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
        "split", type=str, help="input split", choices=["train", "val", "test"]
        )
    parser.add_argument("seg_tag", type=str, help="segmentation identifier")
    parser.add_argument(
        "--phone_tolerance", type=int,
        help="number of frames within which a phone boundary prediction is "
        "still considered correct (default: %(default)s)", default=2)
    parser.add_argument(
        "--word_tolerance", type=int,
        help="number of frames within which a word boundary prediction is "
        "still considered correct (default: %(default)s)", default=2)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def boundaries_to_intervals(boundaries):
    intervals = []
    j_prev = 0
    for j in np.where(boundaries)[0]:
        intervals.append((j_prev, j + 1))
        j_prev = j + 1
    return intervals


def intervals_to_boundaries(intervals):
    boundaries = np.zeros(intervals[-1][1], dtype=bool)
    boundaries[[i[1] - 1 for i in intervals]] = True
    return boundaries


def get_intervals_from_dir(directory, filenames=None):
    interval_dict = {}
    if filenames is None:
        filenames = list(directory.glob("*.txt"))
    else:
        filenames = [
            (Path(directory)/i).with_suffix(".txt") for i in filenames
            ]
    for fn in tqdm(filenames):
        interval_dict[fn.stem] = []
        for i in fn.read_text().strip().split("\n"):
            if len(i) == 0:
                interval_dict.pop(fn.stem)
                continue
            start, end, label = i.split()
            start = int(start)
            end = int(end)
            interval_dict[fn.stem].append((start, end, label))
    return interval_dict


def intervals_to_max_overlap(ref_intervals, pred_intervals, ref_labels=None):
    """
    Each interval is mapped to the reference label with maximum overlap.

    If `ref_labels` is not given, it is assumed that the labels are given as
    the third element of each of the elements in the `ref_intervals` list.
    """
    if ref_labels is None:
        ref_labels = [i[2] for i in ref_intervals]
    mapped_seq = []
    for pred_interval in pred_intervals:
        overlaps = []
        for ref_interval in ref_intervals:
            if ref_interval[1] <= pred_interval[0]:
                overlaps.append(0)
            elif ref_interval[0] >= pred_interval[1]:
                overlaps.append(0)
            else:
                overlap = pred_interval[1] - pred_interval[0]
                if ref_interval[0] > pred_interval[0]:
                    overlap -= (ref_interval[0] - pred_interval[0])
                if ref_interval[1] < pred_interval[1]:
                    overlap -= (pred_interval[1] - ref_interval[1])
                overlaps.append(overlap)
        mapped_seq.append(ref_labels[np.argmax(overlaps)]) 
    return mapped_seq


def str_to_id_labels(d):
    """
    Converts labels given in underscore format to integer IDs.

    For instance, the label "37_18_22_" could be mapped to ID 17. The new
    dictionary as well as the key mappings are returned.

    Return
    ------
    new_dict, str_to_id, id_to_str : (dict, dict, dict)
    """
    label_types = set()
    for key in d:
        for _, _, label in d[key]:
            label_types.add(label)

    key_sorter = lambda x: [int(i) for i in x.split("_") if i != ""]
    str_to_id = {
        s: i for i, s in enumerate(sorted(label_types, key=key_sorter))
        }
    id_to_str = {
        i: s for i, s in enumerate(sorted(label_types, key=key_sorter))
        }

    new_dict = {}
    for key in d:
        new_dict[key] = []
        for start, end, label in d[key]:
            new_dict[key].append((start, end, str_to_id[label]))

    return new_dict, str_to_id, id_to_str


#-----------------------------------------------------------------------------#
#                             EVALUATION FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def score_boundaries(ref, seg, tolerance=0):
    """
    Calculate precision, recall, F-score for the segmentation boundaries.

    Parameters
    ----------
    ref : list of vector of bool
        The ground truth reference.
    seg : list of vector of bool
        The segmentation hypothesis.
    tolerance : int
        The number of slices with which a boundary might differ but still be
        regarded as correct.

    Return
    ------
    output : (float, float, float)
        Precision, recall, F-score.
    """
    n_boundaries_ref = 0
    n_boundaries_seg = 0
    n_boundaries_correct = 0
    for i_boundary, boundary_ref in enumerate(ref):
        boundary_seg = seg[i_boundary]
        assert boundary_ref[-1]  # check if last boundary is True
        assert boundary_seg[-1]
        
        # If lengths are the same, disregard last True reference boundary
        if len(boundary_ref) == len(boundary_seg):
            boundary_ref = boundary_ref[:-1]
            # boundary_seg = boundary_seg[:-1]

        boundary_seg = seg[i_boundary][:-1]  # last boundary is always True,
                                             # don't want to count this

        # If reference is longer, truncate
        if len(boundary_ref) > len(boundary_seg):
            boundary_ref = boundary_ref[:len(boundary_seg)]
        
        boundary_ref = list(np.nonzero(boundary_ref)[0])
        boundary_seg = list(np.nonzero(boundary_seg)[0])
        n_boundaries_ref += len(boundary_ref)
        n_boundaries_seg += len(boundary_seg)

        for i_seg in boundary_seg:
            for i, i_ref in enumerate(boundary_ref):
                if abs(i_seg - i_ref) <= tolerance:
                    n_boundaries_correct += 1
                    boundary_ref.pop(i)
                    break

    # Temp
#     print("n_boundaries_correct", n_boundaries_correct)
#     print("n_boundaries_seg", n_boundaries_seg)
#     print("n_boundaries_ref", n_boundaries_ref)

    precision = float(n_boundaries_correct)/n_boundaries_seg
    recall = float(n_boundaries_correct)/n_boundaries_ref
    if precision + recall != 0:
        f = 2*precision*recall / (precision + recall)
    else:
        f = -np.inf

    return precision, recall, f


def score_word_token_boundaries(ref, seg, tolerance=0):
    """
    Calculate precision, recall, F-score for the word token boundaries.

    Parameters
    ----------
    ref : list of vector of bool
        The ground truth reference.
    seg : list of vector of bool
        The segmentation hypothesis.
    tolerance : int
        The number of slices with which a boundary might differ but still be
        regarded as correct.

    Return
    ------
    output : (float, float, float)
        Precision, recall, F-score.
    """
    n_tokens_ref = 0
    n_tokens_seg = 0
    n_tokens_correct = 0
    for i_boundary, boundary_ref in enumerate(ref):
        boundary_seg = seg[i_boundary]
        assert boundary_ref[-1]  # check if last boundary is True
        assert boundary_seg[-1]
        
        # The code below shouldn't be done for token scores
        # # If lengths are the same, disregard last True reference boundary
        # if len(boundary_ref) == len(boundary_seg):
        #     boundary_ref = boundary_ref[:-1]
        # boundary_seg = seg[i_boundary][:-1]  # last boundary is always True,
                                             # don't want to count this

        # If reference is longer, truncate
        if len(boundary_ref) > len(boundary_seg):
            boundary_ref = boundary_ref[:len(boundary_seg)]
            boundary_ref[-1] = True

        # Build list of ((word_start_lower, word_start_upper), (word_end_lower,
        # word_end_upper))
        word_bound_intervals = []
        for word_start, word_end in boundaries_to_intervals(boundary_ref):
            word_bound_intervals.append((
                (max(0, word_start - tolerance), word_start + tolerance),
                (word_end - tolerance, word_end + tolerance)
                ))
        seg_intervals = boundaries_to_intervals(boundary_seg)

        n_tokens_ref += len(word_bound_intervals)
        n_tokens_seg += len(seg_intervals)

        # Score word token boundaries
        for seg_start, seg_end in seg_intervals:
            # print seg_start, seg_end
            for i_gt_word, (word_start_interval,
                    word_end_interval) in enumerate(word_bound_intervals):
                word_start_lower, word_start_upper = word_start_interval
                word_end_lower, word_end_upper = word_end_interval

                if (word_start_lower <= seg_start <= word_start_upper and
                        word_end_lower <= seg_end <= word_end_upper):
                    n_tokens_correct += 1
                    word_bound_intervals.pop(i_gt_word)  # can't re-use token
                    # print "correct"
                    break

    # # Temp
    # print("n_tokens_correct", n_tokens_correct)
    # print("n_tokens_seg", n_tokens_seg)
    # print("n_tokens_ref", n_tokens_ref)

    precision = float(n_tokens_correct)/n_tokens_seg
    recall = float(n_tokens_correct)/n_tokens_ref
    if precision + recall != 0:
        f = 2*precision*recall / (precision + recall)
    else:
        f = -np.inf

    return precision, recall, f


def get_os(precision, recall):
    """Calculate over segmentation score."""
    if precision == 0:
        return -np.inf
    else:
        return recall/precision - 1


def get_rvalue(precision, recall):
    """Calculate the R-value."""
    os = get_os(precision, recall)
    r1 = np.sqrt((1 - recall)**2 + os**2)
    r2 = (-os + recall - 1)/np.sqrt(2)
    rvalue = 1 - (np.abs(r1) + np.abs(r2))/2
    return rvalue


def score_clusters(ref_interval_dict, pred_interval_dict):
    ref_labels = []
    pred_labels = []
    for utt in ref_interval_dict:
    # for utt in tqdm(ref_interval_dict):
        ref = ref_interval_dict[utt]
        pred = pred_interval_dict[utt]
        ref_labels.extend(intervals_to_max_overlap(ref, pred))
        pred_labels.extend([int(i[2]) for i in pred])
    
    pur = purity(ref_labels, pred_labels)

    h, c, V = metrics.homogeneity_completeness_v_measure(
        ref_labels, pred_labels)
    
    return pur, h, c, V


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    # Directories
    seg_dir = (
        Path("exp")/args.model/args.dataset/args.split/args.seg_tag/"intervals"
        )
    phone_ref_dir = Path("data")/args.dataset/"phone_intervals"
    word_ref_dir = Path("data")/args.dataset/"word_intervals"

    # Read segmentation
    segmentation_interval_dict = {}
    print("Reading: {}".format(seg_dir))
    assert seg_dir.is_dir(), "missing directory: {}".format(seg_dir)
    segmentation_interval_dict = get_intervals_from_dir(seg_dir)
    utterances = segmentation_interval_dict.keys()

    # Read phone reference
    print("Reading: {}".format(phone_ref_dir))
    assert phone_ref_dir.is_dir(), "missing directory: {}".format(
        phone_ref_dir
        )
    phone_ref_interval_dict = get_intervals_from_dir(phone_ref_dir, utterances)

    # Read word reference
    if word_ref_dir.is_dir():
        print("Reading: {}".format(word_ref_dir))
        word_ref_interval_dict = get_intervals_from_dir(
            word_ref_dir, utterances
            )

    # Convert intervals to boundaries
    print("Converting intervals to boundaries:")
    segmentation_boundaries_dict = {}
    for utt_key in tqdm(segmentation_interval_dict):
        segmentation_boundaries_dict[utt_key] = intervals_to_boundaries(
            segmentation_interval_dict[utt_key]
            )
    phone_ref_boundaries_dict = {}
    for utt_key in tqdm(phone_ref_interval_dict):
        phone_ref_boundaries_dict[utt_key] = intervals_to_boundaries(
            phone_ref_interval_dict[utt_key]
            )
    if word_ref_dir.is_dir():
        word_ref_boundaries_dict = {}
        for utt_key in tqdm(word_ref_interval_dict):
            word_ref_boundaries_dict[utt_key] = intervals_to_boundaries(
                word_ref_interval_dict[utt_key]
                )

    # Map e.g. "23_12_" to a unique integer ID e.g. 10
    if ("word" in args.seg_tag and "_" in
            list(segmentation_interval_dict.values())[0][0][-1]):
        segmentation_interval_dict, str_to_id, id_to_str = str_to_id_labels(
            segmentation_interval_dict
            )

    # Evaluate phone boundaries
    reference_list = []
    segmentation_list = []
    for utt_key in phone_ref_boundaries_dict:
        reference_list.append(phone_ref_boundaries_dict[utt_key])
        segmentation_list.append(segmentation_boundaries_dict[utt_key])
    p, r, f = score_boundaries(
        reference_list, segmentation_list, tolerance=args.phone_tolerance
        )

    # Evaluate clustering
    # if not "word" in args.seg_tag:
    # print("Scoring clusters (phone):")
    purity, h, c, V = score_clusters(
        phone_ref_interval_dict, segmentation_interval_dict
        )

    print("-"*(79 - 4))
    print("Phone boundaries:")
    print("Precision: {:.2f}%".format(p*100))
    print("Recall: {:.2f}%".format(r*100))
    print("F-score: {:.2f}%".format(f*100))
    print("OS: {:.2f}%".format(get_os(p, r)*100))
    print("R-value: {:.2f}%".format(get_rvalue(p, r)*100))
    print("-"*(79 - 4))

    # if not "word" in args.seg_tag:
    print("Phone clusters:")
    print("Purity: {:.2f}%".format(purity*100))
    print("Homogeneity: {:.2f}%".format(h*100))
    print("Completeness: {:.2f}%".format(c*100))
    print("V-measure: {:.2f}%".format(V*100))
    print("-"*(79 - 4))

    # Word-level evaluation
    if word_ref_dir.is_dir():

        # Evaluate word boundaries
        reference_list = []
        segmentation_list = []
        for utterance in word_ref_boundaries_dict:
            reference_list.append(word_ref_boundaries_dict[utterance])
            segmentation_list.append(segmentation_boundaries_dict[utterance])
        p, r, f = score_boundaries(
            reference_list, segmentation_list, tolerance=args.word_tolerance
            )

        # Evaluate clustering
        # print("Scoring clusters (word):")
        purity, h, c, V = score_clusters(
            word_ref_interval_dict, segmentation_interval_dict
            )

        print("Word boundaries:")
        print("Precision: {:.2f}%".format(p*100))
        print("Recall: {:.2f}%".format(r*100))
        print("F-score: {:.2f}%".format(f*100))
        print("OS: {:.2f}%".format(get_os(p, r)*100))
        print("R-value: {:.2f}%".format(get_rvalue(p, r)*100))
        print("-"*(79 - 4))

        # Word token boundaries
        p, r, f = score_word_token_boundaries(
            reference_list, segmentation_list, tolerance=args.word_tolerance
            )
        print("Word token boundaries:")
        print("Precision: {:.2f}%".format(p*100))
        print("Recall: {:.2f}%".format(r*100))
        print("F-score: {:.2f}%".format(f*100))
        print("OS: {:.2f}%".format(get_os(p, r)*100))
        # print("R-value: {:.2f}%".format(get_rvalue(p, r)*100))
        print("-"*(79 - 4))

        # if not "word" in args.seg_tag:
        print("Word clusters:")
        print("Purity: {:.2f}%".format(purity*100))
        print("Homogeneity: {:.2f}%".format(h*100))
        print("Completeness: {:.2f}%".format(c*100))
        print("V-measure: {:.2f}%".format(V*100))
        print("-"*(79 - 4))


if __name__ == "__main__":
    main()
