Unsupervised Phone and Word Segmentation using Vector-Quantized Neural Networks
===============================================================================

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](license.md)


**GO THROUGH ALL to-do'S!!!**


## Overview

Unsupervised phone and word segmentation on speech data is performed. The
experiments are described in:

- H. Kamper, "Dynamic programming on self-supervised features for word
  segmentation on discovered phone units," *arXiv preprint arXiv:2202.???*,
  2022. [[arXiv](https://arxiv.org/????)]
- H. Kamper and B. van Niekerk, "Towards unsupervised phone and word
  segmentation using self-supervised vector-quantized neural networks," in
  *Proc. Interspeech*, 2021. [[arXiv](http://arxiv.org/abs/2012.07551)]

Please cite these papers if you use the code.


## Dependencies

Dependencies can be installed in a conda environment:

    conda env create -f environment.yml
    conda activate dpdp

This does not include [wordseg](https://wordseg.readthedocs.io/), which should
be installed in its own environment according to the documentation.

Install the [DPDP AE-RNN](https://github.com/kamperh/dpdp_aernn/) package:

    git clone https://github.com/kamperh/dpdp_aernn.git ../


## Minimal usage example: DPDP AE-RNN with DPDP CPC+K-means on Buckeye

In the sections that follow I give more complete details. In this section I
briefly outline the sequence of steps that should reproduce the DPDP system
results on Buckeye given in [the paper](*to-do*). To apply the approach on
other datasets you will need to carefully work through the subsequent sections,
but I hope that this current section helps you get going.

1. Obtain the ground truth alignments for Buckeye provided as part of [this
   release](*to-do*). These should be extracted so that you have a
   `data/buckeye/` directory in which the alignments are given.

2. Extract CPC+K-means features for Buckeye. Do this by following the steps in
   [this subsection](*to-do*).

3. 


## Dataset format and directory structure

This code should be usable with any dataset given that alignments and VQ
encodings are provided.

The data files for the Buckeye corpus described below can be downloaded as part
of a release at [this link](**To-do.**)

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

Any VQ model can be used. In the section below I give an example of how VQ-VAE,
VQ-CPC and CPC+K-means models can be used to obtain codes for the Buckeye
dataset. In the subsequent section DPDP segmentation is described.


## Example encodings: CPC+K-means features on Buckeye

Install the ZeroSpeech 2021 baseline system from [my
fork](https://github.com/kamperh/zerospeech2021_baseline) by following the
steps in the [installation section of the readme](to-do). Make sure that
`vqwordseg/` (this repository) and `zerospeech2021_baseline/` are in the same
directory, i.e. after cloning you should have a `../zerospeech2021_baseline/`
directory relative to the root that you are currently in.

Move to the ZeroSpeech 2021 directory:

    cd ../zerospeech2021_baseline/

Extract individual Buckeye wav files:

    ./get_buckeye_wavs.py ~/endgame/datasets/buckeye/

Encode the Buckeye dataset:

    conda activate zerospeech2021_baseline
    ./encode.py wav/buckeye/val/ exp/buckeye/val/
    ./encode.py wav/buckeye/test/ exp/buckeye/test/

Move back and deactivate the environment:

    cd ../vqwordseg/
    conda deactivate


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

    ./encode.py checkpoint=checkpoints/cpc/english2019/model.ckpt-22000.pt split=val save_indices=True save_auxiliary=True save_embedding=../vqwordseg/exp/vqcpc/buckeye/embedding.npy out_dir=../vqwordseg/exp/vqcpc/buckeye/val/ dataset=buckeye

Change directory to `../VectorQuantizedVAE` and then run the following there.
The audio can be pre-processed again (as above), or alternatively you can
simply link to the audio from `VectorQuantizedCPC/`:

    ln -s ../VectorQuantizedCPC/datasets/ .

Encode the data and write it to the `vqwordseg/exp/` directory. This should
be performed for all splits (`train`, `val` and `test`):

    # Buckeye
    ./encode.py checkpoint=checkpoints/2019english/model.ckpt-500000.pt split=train save_indices=True save_auxiliary=True save_embedding=../vqwordseg/exp/vqvae/buckeye/embedding.npy out_dir=../vqwordseg/exp/vqvae/buckeye/train/ dataset=buckeye

You can delete all the created `auxiliary_embedding1/` and `codes/` directories
since these are not used for segmentation.


## Phone segmentation

DP penalized segmentation:

    # Buckeye (GMM)
    ./vq_phoneseg.py --downsample_factor 1 --input_format=npy --algorithm=dp_penalized --dur_weight 0.001 gmm buckeye val --output_tag phoneseg_merge

    # Buckeye (VQ-CPC)
    ./vq_phoneseg.py --input_format=txt --algorithm=dp_penalized vqcpc buckeye val

    # Buckeye (VQ-VAE)
    ./vq_phoneseg.py vqvae buckeye val

    # Buckeye (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt --algorithm=dp_penalized cpc_big buckeye val

    # Buckeye (CPC-big) HSMM
    ./vq_phoneseg.py --algorithm dp_penalized_hsmm --downsample_factor 1 --dur_weight 1.0 --model_eos --dur_weight_func neg_log_gamma --output_tag=phoneseg_hsmm_tune cpc_big buckeye val

    # Buckeye Felix split (CPC-big) HSMM
    ./vq_phoneseg.py --algorithm dp_penalized_hsmm --downsample_factor 1 --dur_weight 1.0 --model_eos --dur_weight_func neg_log_gamma --output_tag=phoneseg_hsmm_tune cpc_big buckeye_felix test

    # Xitsonga (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt --algorithm=dp_penalized cpc_big xitsonga train

    # Buckeye (XLSR)
    ./vq_phoneseg.py --downsample_factor 2 --dur_weight 2500 --input_format=npy --algorithm=dp_penalized xlsr buckeye val

    # Buckeye (ResDAVEnet-VQ)
    ./vq_phoneseg.py --downsample_factor 2 --dur_weight 3 --input_format=txt --algorithm=dp_penalized resdavenet_vq buckeye val

    # Buckeye (ResDAVEnet-VQ3)
    ./vq_phoneseg.py --downsample_factor 4 --dur_weight 0.001 --input_format=txt --algorithm=dp_penalized resdavenet_vq_quant3 buckeye val --output_tag=phoneseg_merge

    # Buckeye Felix split (VQ-VAE)
    ./vq_phoneseg.py --output_tag=phoneseg_dp_penalized vqvae buckeye_felix test

    # Buckeye Felix split (CPC-big)
    ./vq_phoneseg.py  --downsample_factor 1 --dur_weight 2 --output_tag=phoneseg_dp_penalized_tune cpc_big buckeye_felix val

    # Buckeye Felix split (VQ-VAE) with Poisson duration prior
    ./vq_phoneseg.py --output_tag=phoneseg_dp_penalized_poisson --dur_weight_func neg_log_poisson --dur_weight 2 vqvae buckeye_felix val

    # Buckeye (VQ-VAE) with Gamma duration prior
    ./vq_phoneseg.py --output_tag=phoneseg_dp_penalized_gamma --dur_weight_func neg_log_gamma --dur_weight 15 vqvae buckeye val

    # ZeroSpeech'17 English (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt --algorithm=dp_penalized cpc_big zs2017_en train

    # ZeroSpeech'17 French (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt --algorithm=dp_penalized cpc_big zs2017_fr train

    # ZeroSpeech'17 Mandarin (CPC-big)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 2 --input_format=txt --algorithm=dp_penalized cpc_big zs2017_zh train

    # ZeroSpeech'17 French (XLSR)
    ./vq_phoneseg.py --downsample_factor 2 --dur_weight 1500 --input_format=npy --algorithm=dp_penalized xlsr zs2017_fr train

    # ZeroSpeech'17 Mandarin (XLSR)
    ./vq_phoneseg.py --downsample_factor 2 --dur_weight 2500 --input_format=npy --algorithm=dp_penalized xlsr zs2017_zh train

DP penalized N-seg. segmentation:

    # Buckeye Felix split (VQ-VAE)
    ./vq_phoneseg.py --algorithm=dp_penalized_n_seg --n_frames_per_segment=3 --n_min_segments=3 vqvae buckeye_felix test

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
    ./vq_wordseg.py --dur_weight=6 --algorithm=dpdp_aernn gmm buckeye val phoneseg_dp_penalized

    # Buckeye (CPC-big)
    ./vq_wordseg.py --algorithm=dpdp_aernn cpc_big buckeye val phoneseg_dp_penalized

Evaluate the segmentation:

    # Buckeye (VQ-VAE)
    ./eval_segmentation.py vqvae buckeye val wordseg_ag_dp_penalized

    # Buckeye (CPC-big)
    ./eval_segmentation.py cpc_big buckeye val wordseg_ag_dp_penalized

Evaluate the segmentation with the ZeroSpeech tools:

    ./intervals_to_zs.py cpc_big zs2017_zh train wordseg_segaernn_dp_penalized
    cd ../zerospeech2017_eval/
    ln -s /media/kamperh/endgame/projects/stellenbosch/vqseg/vqwordseg/exp/cpc_big/zs2017_zh/train/wordseg_segaernn_dp_penalized/clusters.txt 2017/track2/mandarin.txt
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

    ./indices_to_txt.py vqvae buckeye val phoneseg_dp_penalized s18_03a_025476-025541
    cd ../VectorQuantizedVAE
    ./synthesize_codes.py checkpoints/2019english/model.ckpt-500000.pt ../vqwordseg/s18_03a_025476-025541.txt
    cd -


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


## Old work-flow

1. Extract CPC+K-means features in `../zerospeech2021_baseline/`.
2. Perform phone segmentation here using `vq_phoneseg.py`.
3. Move to `../seg_aernn/notebooks/` and perform word segmentation.
4. Move back here and evaluate the segmentation using `eval_segmentation.py`.
5. For ZeroSpeech systems, the evaluation is done in `../zerospeech2017_eval/`.


## Disclaimer

The code provided here is not pretty. But research should be reproducible. I
provide no guarantees with the code, but please let me know if you have any
problems, find bugs or have general comments.
