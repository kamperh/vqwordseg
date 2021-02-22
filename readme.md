Phone and Word Segmentation using Vector-Quantised Neural Networks
==================================================================

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](license.md)

**To-do.** Maybe GPL-v3? Then need to update start of notebooks as well.


Disclaimer
----------
The code provided here is not pretty. But research should be reproducible. I
provide no guarantees with the code, but please let me know if you have any
problems, find bugs or have general comments.


Installation
------------
You will require the following:

- [Python 3](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/)
- [tqdm](https://tqdm.github.io/)
- [scikit-learn](https://scikit-learn.org/)
- [wordseg](https://wordseg.readthedocs.io/)

**To-do.** Add and describe a conda installation file.


Minimal example
---------------
**To-do.** Maybe a Colab notebook in which everything is installed and VQ-VAE
codes are extracted for an input utterance. Synthesis can maybe even be
performed.


Dataset format and directory structure
--------------------------------------
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

The VQ-segmentation algorithms operate on the output of VQ models. The
(pre-)quantized representations and code indices should be provided in the
`exp/` directory. These are used as input to the VQ-segmentation algorithms;
the segmented output is also produced in `exp/`.

As an example, the directory `exp/vqcpc/buckeye/` should contain a file
`embedding.npy`, which is the codebook matrix for a
[VQ-CPC](https://github.com/kamperh/VectorQuantizedCPC) model trained on
Buckeye. This matrix will have the shape `[n_codes, code_dim]`. The directory
`exp/vqcpc/buckeye/val/` needs to contain at least subdirectories for the
encoded validation set:

- `auxiliary_embedding2/`
- `indices/`

The `auxiliary_embedding2/` directory contains the encodings from the VQ model
before quantization. These encodings are given as text files with an embedding
per line, e.g. the first three lines of
`auxiliary_embedding2/s01_01a_003222-003256.txt` could be:

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

Any VQ model can be used. In the section below I give an example of how VQ-VAE
and VQ-CPC models can be used to obtain codes for the Buckeye dataset. In the
subsequent section VQ-segmentation is described.


Example encodings: VQ-VAE and VQ-CPC input representations
----------------------------------------------------------
You can obtain the VQ input representations using the file format indicated
above. As an example, here I describe how I did it for the Buckeye data. The
data files for the Buckeye corpus can be downloaded as part of a release at
[this link](**To-do.**)

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


Phone segmentation
------------------
DP penalized segmentation:

    # Buckeye (VQ-CPC)
    ./vq_phoneseg.py --input_format=txt --algorithm=dp_penalized vqcpc buckeye val

    # Buckeye (VQ-VAE)
    ./vq_phoneseg.py vqvae buckeye val

    # Buckeye (Big CPC)
    ./vq_phoneseg.py --downsample_factor 1 --dur_weight 3 --input_format=txt --algorithm=dp_penalized cpc_big buckeye val

    # Buckeye Felix split (VQ-VAE)
    ./vq_phoneseg.py --output_tag=phoneseg_dp_penalized vqvae buckeye_felix test

    # Buckeye Felix split (VQ-VAE) with Poisson duration prior
    ./vq_phoneseg.py --output_tag=phoneseg_dp_penalized_poisson --dur_weight_func neg_log_poisson --dur_weight 2 vqvae buckeye_felix val

    # Buckeye (VQ-VAE) with Gamma duration prior
    ./vq_phoneseg.py --output_tag=phoneseg_dp_penalized_gamma --dur_weight_func neg_log_gamma --dur_weight 15 vqvae buckeye val

DP penalized N-seg. segmentation:

    # Buckeye Felix split (VQ-VAE)
    ./vq_phoneseg.py --algorithm=dp_penalized_n_seg --n_frames_per_segment=3 --n_min_segments=3 vqvae buckeye_felix test

Evaluate segmentation:

    # Buckeye (VQ-VAE)
    ./eval_segmentation.py vqvae buckeye val phoneseg_dp_penalized_n_seg


Word segmentation
-----------------
Word segmentation are performed on the segmented phone sequences.

Adaptor grammar word segmentation:

    conda activate wordseg
    ./vq_wordseg.py --algorithm=ag vqvae buckeye val phoneseg_dp_penalized

Evaluate the segmentation:

    ./eval_segmentation.py vqvae buckeye val wordseg_ag_dp_penalized


Analysis
--------
Listen to segmented codes:

    ./cluster_wav.py vqvae buckeye val phoneseg_dp_penalized 343
    ./cluster_wav.py vqvae buckeye val wordseg_tp_dp_penalized 486_

This requires `sox` and that you change the path at the beginning of
`cluster_wav.py`.

Synthesize an utterance:

    ./indices_to_txt.py vqvae buckeye val phoneseg_dp_penalized s18_03a_025476-025541
    cd ../VectorQuantizedVAE
    ./synthesize_codes.py checkpoints/2019english/model.ckpt-500000.pt ../vqwordseg/s18_03a_025476-025541.txt
    cd -
