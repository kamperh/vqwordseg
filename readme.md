Phone and Word Segmentation using Vector-Quantised Neural Networks
==================================================================

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](license.md)


Disclaimer
----------
The code provided here is not pretty. But I believe that research should be
reproducible. I provide no guarantees with the code, but please let me know if
you have any problems, find bugs or have general comments.


Installation
------------
You will require the following:

- [Python 3](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/)
- [tqdm](https://tqdm.github.io/)
- [scikit-learn](https://scikit-learn.org/)
- [wordseg](https://wordseg.readthedocs.io/)
- [VectorQuantizedVAE fork (ZeroSpeech)](https://github.com/kamperh/ZeroSpeech)
- [VectorQuantizedCPC fork](https://github.com/kamperh/VectorQuantizedCPC)

Make sure all dependencies for `VectorQuantizedVAE` and `VectorQuantizedCPC`
are also satisfied. To install these packages locally, run
`./install_local.sh`.

**To-do.** Add and describe a conda installation file.


Minimal example
---------------
**To-do.** Maybe a Colab notebook in which everything is installed and VQ-VAE
codes are extracted for an input utterance. Synthesis can maybe even be
performed.


Data and directory structure
----------------------------
For evaluation you need the ground truth phone (and optionally) word
boundaries. This should be stored in the directories
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

**To-do.** Maybe include download link for Buckeye data files.

The VQ-segmentation algorithms operate on the output of vector quantized neural
networks. The quantized input representations as well as the segmented output
are stored in the experiments directory `exp/`. In the section below I describe
how to obtain the inputs and in the subsequent sections how to perform
segmentation.

As an example, the directory `exp/vqcpc/buckeye/` should contain a file
`embedding.npy`, which is the codebook matrix for the VQ-CPC trained on
Buckeye. The directory `exp/vqcpc/buckeye/val/` needs to contain three
subdirectories for the processed validation set:

- `auxiliary_embedding2/`
- `codes/`
- `indices/`

The codes and auxiliary embeddings will have an embedding per line, e.g. the
first two lines of `auxiliary_embedding2/s01_01a_003222-003256.txt`:

    0.1601707935333252 -0.0403369292616844 0.4687763750553131 ...
    0.4489639401435852  1.3353070020675659 1.0353083610534668 ...

The indices should be the code indices, each index on a new line.


VQ-VAE and VQ-CPC input representations
---------------------------------------
You can obtain the VQ input representations using the file format indicated
above. As an example, here I describe how I did it for the Buckeye data.

Change directory to `../VectorQuantizedCPC` and then perform the following
steps there. Pre-process audio and extract log-Mel spectrograms:

    ./preprocess.py in_dir=../datasets/buckeye/ dataset=buckeye

Encode the data and write it to the `vqwordseg/exp/` directory. This should be
performed for all splits (`train`, `val` and `test`):

    ./encode.py checkpoint=checkpoints/cpc/english2019/model.ckpt-22000.pt split=val save_indices=True save_auxiliary=True save_embedding=../vqwordseg/exp/vqcpc/buckeye/embedding.npy out_dir=../vqwordseg/exp/vqcpc/buckeye/val/ dataset=buckeye

Change directory to `../VectorQuantizedVAE` and then run the following there.
The audio can be pre-processed again (as above), or alternatively you can
simply link to the audio from `VectorQuantizedCPC`:

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

    # Buckeye Felix split (VQ-VAE)
    ./vq_phoneseg.py --output_tag=phoneseg_dp_penalized vqvae buckeye_felix test

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

