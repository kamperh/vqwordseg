Unsupervised Phone and Word Segmentation using Vector-Quantized Neural Networks
===============================================================================

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](license.md)


## Overview

Unsupervised phone and word segmentation on speech data is performed. The
experiments are described in:

- H. Kamper, "Word segmentation on discovered phone units with dynamic
  programming and self-supervised scoring," *IEEE/ACM Transactions on Audio,
  Speech and Language Processing*, vol. 31, pp. 684-694, 2023.
  [[arXiv](https://arxiv.org/abs/2202.11929)]
- H. Kamper and B. van Niekerk, "Towards unsupervised phone and word
  segmentation using self-supervised vector-quantized neural networks," in
  *Proc. Interspeech*, 2021. [[arXiv](http://arxiv.org/abs/2012.07551)]

Please cite these papers if you use the code.


## Dependencies

Dependencies can be installed in a conda environment:

    conda env create -f environment.yml
    conda activate dpdp

This does not include [wordseg](https://wordseg.readthedocs.io/), which should
be installed in its own environment according to its documentation.

Install the [DPDP AE-RNN](https://github.com/kamperh/dpdp_aernn/) package:

    git clone https://github.com/kamperh/dpdp_aernn.git ../dpdp_aernn


## Minimal usage example: DPDP AE-RNN with DPDP CPC+K-means on Buckeye

In the sections that follow I give more complete details. In this section I
briefly outline the sequence of steps that should reproduce the DPDP system
results on Buckeye given in [the paper](https://arxiv.org/abs/2202.11929). To
apply the approach on other datasets you will need to carefully work through
the subsequent sections, but I hope that this current section helps you to get
going.

1.  Obtain the ground truth alignments for Buckeye provided in
    [buckeye.zip](https://github.com/kamperh/vqwordseg/releases/download/v1.0/buckeye.zip)
    as part of [this
    release](https://github.com/kamperh/vqwordseg/releases/tag/v1.0). Extract
    it into `data/`. There should now be a `data/buckeye/` directory with the
    alignments.

2.  Extract CPC+K-means features for Buckeye. Do this by following the steps in
    [the CPC-big subsection](#example-encodings-cpc-big-features-on-buckeye)
    below.

3.  Perform acoustic unit discovery using DPDP CPC+K-means:

        ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 \
            --input_format=txt --algorithm=dp_penalized cpc_big buckeye val

4.  Perform word segmentation on the discovered units using the DPDP AE-RNN:

        ./vq_wordseg.py --algorithm=dpdp_aernn \
            cpc_big buckeye val phoneseg_dp_penalized

5.  Evaluate the segmentation:

        ./eval_segmentation.py cpc_big buckeye val \
            wordseg_dpdp_aernn_dp_penalized

The result should correspond approximately to the following on the Buckeye
validation data:

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 35.80%
    Recall: 36.30%
    F-score: 36.05%
    OS: 1.40%
    R-value: 45.13%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 23.93%
    Recall: 24.23%
    F-score: 24.08%
    OS: 1.24%
    ---------------------------------------------------------------------------


## Example encodings: CPC-big features on Buckeye

Install the ZeroSpeech 2021 baseline system from [my
fork](https://github.com/kamperh/zerospeech2021_baseline) by following the
steps in the [installation section of the
readme](https://github.com/kamperh/zerospeech2021_baseline#Installation). Make
sure that `vqwordseg/` (this repository) and `zerospeech2021_baseline/` are in
the same directory.

From the `vqwordseg/` directory, move to the ZeroSpeech 2021 directory:

    cd ../zerospeech2021_baseline/

Extract individual Buckeye wav files:

    ./get_buckeye_wavs.py ../datasets/buckeye/

The argument should point to your local copy of Buckeye.

Encode the Buckeye data:

    conda activate zerospeech2021_baseline
    ./encode.py wav/buckeye/val/ exp/buckeye/val/
    ./encode.py wav/buckeye/test/ exp/buckeye/test/

    conda activate dpdp
    ./encode_hubert.py wav/buckeye/val/ exp/buckeye/hubert/val/
    ./encode_hubert.py wav/buckeye/test/ exp/buckeye/hubert/test/

Move back and deactivate the environment:

    cd ../vqwordseg/
    conda activate dpdp


## Dataset format and directory structure

This code should be usable with any dataset given that alignments and VQ
encodings are provided.

For evaluation you need the ground truth phone and (optionally) word
boundaries. These should be stored in the directories
`data/<dataset>/phone_intervals/` and `data/<dataset>/word_intervals/` using
the following filename format:

    <speaker>_<utterance_id>_<start_frame>-<end_frame>.txt

E.g., `data/buckeye/phone_intervals/s01_01a_003222-003256.txt` could consist
of:

    0 5 hh
    5 10 iy
    10 15 jh
    15 19 ih
    19 27 s
    27 34 s
    34 46 iy
    46 54 m
    54 65 z
    65 69 l
    69 78 ay
    78 88 k

The duration-penalized dynamic programming (DPDP) algorithms operate on the
output vector quantized (VQ) models. The (pre-)quantized representations and
code indices should be provided in the `exp/` directory. These are used as
input to the VQ-segmentation algorithms; the segmented output is also produced
in `exp/`.

As an example, the directory `exp/vqcpc/buckeye/` should contain a file
`embedding.npy`, which is the codebook matrix for a
[VQ-CPC](https://github.com/kamperh/VectorQuantizedCPC) model trained on
Buckeye. This matrix will have the shape `[n_codes, code_dim]`. The directory
`exp/vqcpc/buckeye/val/` needs to contain at least subdirectories for the
encoded validation set:

- `prequant/`
- `indices/`

The `prequant/` directory contains the encodings from the VQ model before
quantization. These encodings are given as text files with an embedding per
line, e.g. the first three lines of `prequant/s01_01a_003222-003256.txt` could
be:

     0.1601707935333252 -0.0403369292616844  0.4687763750553131 ...
     0.4489639401435852  1.3353070020675659  1.0353083610534668 ...
    -1.0552909374237061  0.6382007002830505  4.5256714820861816 ...


The `indices/` directory contains the code indices to which the auxiliary
embeddings are actually mapped, i.e. which of the codes in `embedding.npy` are
closest (under some metric) to the pre-quantized embedding. The code indices
are again given as text files, with each index on a new line, e.g. the first
three lines of `indices/s01_01a_003222-003256.txt` could be:

    423
    381
    119
    ...

Any VQ model can be used. In the preceding section section I gave an example of
using CPC-big with K-means; in the section below I give an example of how
VQ-VAE and VQ-CPC can be used to obtain codes for the Buckeye dataset. In the
subsequent section DPDP segmentation is described.


## Example encodings: VQ-VAE and VQ-CPC on Buckeye

You can obtain the VQ input representations using the file format indicated
above. As an example, here I describe how I did it for the Buckeye data.

First the following repositories need to be installed with their dependencies:

- [VectorQuantizedVAE fork (ZeroSpeech)](https://github.com/kamperh/ZeroSpeech)
- [VectorQuantizedCPC fork](https://github.com/kamperh/VectorQuantizedCPC)

If you made sure that the dependencies are satisfied, these packages can be
installed locally by running `./install_local.sh`.

Change directory to `../VectorQuantizedCPC` and then perform the following
steps there. Pre-process audio and extract log-Mel spectrograms:

    ./preprocess.py in_dir=../datasets/buckeye/ dataset=buckeye

Encode the data and write it to the `vqwordseg/exp/` directory. This should be
performed for all splits (`train`, `val` and `test`):

    ./encode.py checkpoint=checkpoints/cpc/english2019/model.ckpt-22000.pt \
        split=val \
        save_indices=True \
        save_auxiliary=True \
        save_embedding=../vqwordseg/exp/vqcpc/buckeye/embedding.npy \
        out_dir=../vqwordseg/exp/vqcpc/buckeye/val/ \
        dataset=buckeye

Change directory to `../VectorQuantizedVAE` and then run the following there.
The audio can be pre-processed again (as above), or alternatively you can
simply link to the audio from `VectorQuantizedCPC/`:

    ln -s ../VectorQuantizedCPC/datasets/ .

Encode the data and write it to the `vqwordseg/exp/` directory. This should be
performed for all splits (`train`, `val` and `test`):

    # Buckeye
    ./encode.py checkpoint=checkpoints/2019english/model.ckpt-500000.pt \
        split=train \
        save_indices=True \
        save_auxiliary=True \
        save_embedding=../vqwordseg/exp/vqvae/buckeye/embedding.npy \
        out_dir=../vqwordseg/exp/vqvae/buckeye/train/ \
        dataset=buckeye

You can delete all the created `auxiliary_embedding1/` and `codes/` directories
since these are not used for segmentation.


## Phone segmentation

DP penalized segmentation:

    # Buckeye (GMM)
    ./vq_phoneseg.py --downsample_factor 1 --input_format=npy \
        --algorithm=dp_penalized --dur_weight 0.001 \
        gmm buckeye val --output_tag phoneseg_merge

    # Buckeye (VQ-CPC)
    ./vq_phoneseg.py --input_format=txt --algorithm=dp_penalized \
        vqcpc buckeye val

    # Buckeye (VQ-VAE)
    ./vq_phoneseg.py vqvae buckeye val

    # Buckeye (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt \
        --algorithm=dp_penalized cpc_big buckeye val

    # Buckeye (HuBERT)
    ./vq_phoneseg.py --downsample_factor 2 --dur_weight 3 --input_format=npy \
        --algorithm=dp_penalized hubert buckeye val

    # Buckeye (CPC-big) HSMM
    ./vq_phoneseg.py --algorithm dp_penalized_hsmm --downsample_factor 1 \
        --dur_weight 1.0 --model_eos --dur_weight_func neg_log_gamma \
        --output_tag=phoneseg_hsmm_tune cpc_big buckeye val

    # Buckeye Felix split (CPC-big) HSMM
    ./vq_phoneseg.py --algorithm dp_penalized_hsmm --downsample_factor 1 \
        --dur_weight 1.0 --model_eos --dur_weight_func neg_log_gamma \
        --output_tag=phoneseg_hsmm_tune cpc_big buckeye_felix test

    # Xitsonga (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt \
        --algorithm=dp_penalized cpc_big xitsonga train

    # Buckeye (XLSR)
    ./vq_phoneseg.py --downsample_factor 2 --dur_weight 2500 \
        --input_format=npy --algorithm=dp_penalized xlsr buckeye val

    # Buckeye (ResDAVEnet-VQ)
    ./vq_phoneseg.py --downsample_factor 2 --dur_weight 3 --input_format=txt \
        --algorithm=dp_penalized resdavenet_vq buckeye val

    # Buckeye (ResDAVEnet-VQ3)
    ./vq_phoneseg.py --downsample_factor 4 --dur_weight 0.001 \
        --input_format=txt --algorithm=dp_penalized resdavenet_vq_quant3 \
        buckeye val --output_tag=phoneseg_merge

    # Buckeye Felix split (VQ-VAE)
    ./vq_phoneseg.py --output_tag=phoneseg_dp_penalized \
        vqvae buckeye_felix test

    # Buckeye Felix split (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 \
        --output_tag=phoneseg_dp_penalized_tune cpc_big buckeye_felix val

    # Buckeye Felix split (VQ-VAE) with Poisson duration prior
    ./vq_phoneseg.py --output_tag=phoneseg_dp_penalized_poisson \
        --dur_weight_func neg_log_poisson --dur_weight 2 \
        vqvae buckeye_felix val

    # Buckeye (VQ-VAE) with Gamma duration prior
    ./vq_phoneseg.py --output_tag=phoneseg_dp_penalized_gamma \
        --dur_weight_func neg_log_gamma --dur_weight 15 vqvae buckeye val

    # ZeroSpeech'17 English (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt \
        --algorithm=dp_penalized cpc_big zs2017_en train

    # ZeroSpeech'17 French (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt \
        --algorithm=dp_penalized cpc_big zs2017_fr train

    # ZeroSpeech'17 Mandarin (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt \
        --algorithm=dp_penalized cpc_big zs2017_zh train

    # ZeroSpeech'17 French (XLSR)
    ./vq_phoneseg.py --downsample_factor 2 --dur_weight 1500 \
        --input_format=npy --algorithm=dp_penalized xlsr zs2017_fr train

    # ZeroSpeech'17 Mandarin (XLSR)
    ./vq_phoneseg.py --downsample_factor 2 --dur_weight 2500 \
        --input_format=npy --algorithm=dp_penalized xlsr zs2017_zh train

    # ZeroSpeech'17 Lang2 (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt \
        --algorithm=dp_penalized cpc_big zs2017_lang2 train

DP penalized N-seg. segmentation:

    # Buckeye Felix split (VQ-VAE)
    ./vq_phoneseg.py --algorithm=dp_penalized_n_seg \
        --n_frames_per_segment=3 --n_min_segments=3 vqvae buckeye_felix test

Evaluate segmentation:

    # Buckeye (VQ-VAE)
    ./eval_segmentation.py vqvae buckeye val phoneseg_dp_penalized_n_seg

    # Buckeye (CPC-big)
    ./eval_segmentation.py cpc_big buckeye val phoneseg_dp_penalized


## Word segmentation

Word segmentation are performed on the segmented phone sequences.

Adaptor grammar word segmentation:

    conda activate wordseg
    # Buckeye (VQ-VAE)
    ./vq_wordseg.py --algorithm=ag vqvae buckeye val phoneseg_dp_penalized

    # Buckeye (CPC-big)
    ./vq_wordseg.py --algorithm=ag cpc_big buckeye val phoneseg_dp_penalized

DPDP AE-RNN word segmentation:

    # Buckeye (GMM)
    ./vq_wordseg.py --dur_weight=6 --algorithm=dpdp_aernn \
        gmm buckeye val phoneseg_dp_penalized

    # Buckeye (CPC-big)
    ./vq_wordseg.py --algorithm=dpdp_aernn \
        cpc_big buckeye val phoneseg_dp_penalized

    # Buckeye (HuBERT)
    ./vq_wordseg.py --dur_weight 6 --algorithm=dpdp_aernn \
        hubert buckeye val phoneseg_dp_penalized

DPDP AE-RNN word segmentation followed by K-means clustering on the AE-RNN
embeddings:

    # Buckeye (CPC-big)
    ./vq_wordseg.py --kmeans 14000 --algorithm=dpdp_aernn \
        cpc_big buckeye val phoneseg_dp_penalized

Perform K-means clustering on top of averaged features:

    # Buckeye (HuBERT)
    ./vq_lexicon_avgembed.py --kmeans 14000 hubert buckeye val wordseg_dpdp_aernn_dp_penalized

Evaluate the segmentation:

    # Buckeye (VQ-VAE)
    ./eval_segmentation.py vqvae buckeye val wordseg_ag_dp_penalized

    # Buckeye (CPC-big)
    ./eval_segmentation.py cpc_big buckeye val wordseg_ag_dp_penalized

    # Buckeye (HuBERT)
    ./eval_segmentation.py hubert buckeye val wordseg_dpdp_aernn_dp_penalized

    # Buckeye (HuBERT) with averaged embeddings
    ./eval_segmentation.py hubert buckeye val wordseg_dpdp_avgembed_kmeans14000

Evaluate the segmentation with the ZeroSpeech tools:

    ./intervals_to_zs.py cpc_big zs2017_zh train wordseg_segaernn_dp_penalized
    cd ../zerospeech2017_eval/
    ln -s \
        ~/endgame/projects/stellenbosch/vqseg/vqwordseg/exp/cpc_big/zs2017_zh/train/wordseg_dpdp_aernn_dp_penalized/clusters.txt \
        2017/track2/mandarin.txt
    conda activate zerospeech2020_updated
    zerospeech2020-evaluate 2017-track2 . -l mandarin -o mandarin.json


## Analysis

Print the word clusters:

    ./clusters_print.py cpc_big buckeye val wordseg_ag_dp_penalized

Listen to segmented codes:

    ./cluster_wav.py vqvae buckeye val phoneseg_dp_penalized 343
    ./cluster_wav.py vqvae buckeye val wordseg_tp_dp_penalized 486_
    ./cluster_wav.py cpc_big buckeye val phoneseg_dp_penalized 50

This requires `sox` and that you change the path at the beginning of
`cluster_wav.py`. For ZeroSpeech'17 data, use `cluster_wav_zs2017.py` instead.

Synthesize an utterance:

    ./indices_to_txt.py vqvae buckeye val phoneseg_dp_penalized \
        s18_03a_025476-025541
    cd ../VectorQuantizedVAE
    ./synthesize_codes.py checkpoints/2019english/model.ckpt-500000.pt \
        ../vqwordseg/s18_03a_025476-025541.txt
    cd -


## Complete example on ZeroSpeech data

An example of phone and word segmentation on the surprise language.

Encode data:

    cd ../zerospeech2021_baseline
    conda activate pytorch
    ./get_wavs.py path_to_data/datasets/zerospeech2020/2020/2017/ \
        zs2017_lang1 train

    conda activate zerospeech2021_baseline
    ./encode.py wav/zs2017_lang1/train/ exp/zs2017_lang1/train/

Phone segmentation:

    cd ../vqwordseg
    conda activate pytorch
    # Create links in exp/cpc_big/
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt \
        --algorithm=dp_penalized cpc_big zs2017_lang1 train
    ./cluster_wav_zs2017.py cpc_big zs2017_lang1 train phoneseg_dp_penalized 3

Word segmentation:

    ./vq_wordseg.py --algorithm=dpdp_aernn cpc_big zs2017_lang1 train \
        phoneseg_dp_penalized
    ./cluster_wav_zs2017.py cpc_big zs2017_lang1 train \
        wordseg_dpdp_aernn_dp_penalized 33_10_11_14_1_34_

Convert to ZeroSpeech format:

    ./intervals_to_zs.py cpc_big zs2017_lang1 train \
        wordseg_dpdp_aernn_dp_penalized

Evaluate the segmentation with the ZeroSpeech tools:

    ./intervals_to_zs.py cpc_big zs2017_zh train \
        wordseg_dpdp_aernn_dp_penalized
    cd ../zerospeech2017_eval/
    conda activate zerospeech2020
    ln -s \
        ~/endgame/projects/stellenbosch/vqseg/vqwordseg/exp/cpc_big/zs2017_zh/train/wordseg_dpdp_aernn_dp_penalized/clusters.txt \
        2017/track2/mandarin.txt
    zerospeech2020-evaluate 2017-track2 . -l mandarin -o mandarin.json


## About the Buckeye data splits

The particular split of Buckeye that I use in this repository is a legacy split
with a somewhat complicated history. But in short the test set is exactly the
same one used in the [ZeroSpeech 2015 challenge](https://zerospeech.com/2015).
The remaining speakers were then used for a validation set and an additional
held-out test set. This additional test set has the same number of speakers as
the validation set, but most papers just report results on the ZeroSpeech 2105
test set.

The result is the following split of Buckeye, according to speaker:

- Train (English1 in [my thesis](https://arxiv.org/abs/1701.00851), devpart1 in
  other repos): s02, s03, s04, s05, s06, s08, s10, s11, s12, s13, s16, s38.
- Validation (devpart2 in other repos): s17, s18, s19, s22, s34, s37, s39, s40.
- Test (English2 in [my thesis](https://arxiv.org/abs/1701.00851), ZS in other
  repos): s01, s20, s23, s24, s25, s26, s27, s29, s30, s31, s32, s33.
- Additional test: s07, s09, s14, s15, s21, s28, s35, s36.

I fist used this in ([Kamper et al., 2017](http://arxiv.org/abs/1606.06950))
and since then in a number of follow-up papers. Others have also used this
split, e.g. ([Drexler and Glass,
2017](https://groups.csail.mit.edu/sls/publications/2017/GLU17_Drexler.pdf)),
([Bhati et al., 2021](https://arxiv.org/abs/2106.02170)), and ([Peng and
Harwath, 2022]https://arxiv.org/abs/2203.15081)).

**Sets used in this repo.** In this repo I only make use of the validation and
test sets above, although features are extracted for the training set. See the
experimental setup section of [the paper](https://arxiv.org/abs/2202.11929).

**The Kreuk split.** Note that [Kreuk et al.
(2020)](https://arxiv.org/abs/2007.13465) uses a different split which is also
used by others. So in the section in [the
paper](https://arxiv.org/abs/2202.11929) where I compare to their approach, I
use their split:

- Train: All Buckeye speakers not below.
- Validation: s25, s36, s39, s40.
- Test: s03, s07, s31, s34.

This split is not included in this repository---it made things too cluttered.
And note that in the [the paper](https://arxiv.org/abs/2202.11929) I again
don't use the Kreuk training set: I only report results on the test data when
comparing to their models.


## Reducing a codebook using clustering

If a codebook is very large, the codes could be reduced by clustering.
The reduced codebook should be saved in a new model directory, and links to
the original pre-quantized features should be created.

As an example, in [cluster_codebook.ipynb](notebooks/cluster_codebook.ipynb),
the ResDAVEnet-VQ codebook is loaded and reduced to 50 codes. The original
codebook had 1024 codes, but only 498 of these were actually used; these are
reduced to 50. The resulting codebook is saved to
`exp/resdavenet_vq_clust50/buckeye/embedding.npy`. The pre-quantized features
are linked to the original version in `exp/resdavenet_vq/`. The indices from
the original model shouldn't be linked, since these doesn't match the new
codebook (but an indices file isn't necessary for running many of the phone
segmentation algorithms).


## Old work-flow (deprecated)

1. Extract CPC+K-means features in `../zerospeech2021_baseline/`.
2. Perform phone segmentation here using `vq_phoneseg.py`.
3. Move to `../seg_aernn/notebooks/` and perform word segmentation.
4. Move back here and evaluate the segmentation using `eval_segmentation.py`.
5. For ZeroSpeech systems, the evaluation is done in `../zerospeech2017_eval/`.


## Disclaimer

The code provided here is not pretty. But research should be reproducible. I
provide no guarantees with the code, but please let me know if you have any
problems, find bugs or have general comments.
