Phone and Word Segmentation Results
===================================

Buckeye
-------

VQ-VAE DP penalized val:

    # ./eval_segmentation.py vqvae buckeye val phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 66.42%
    Recall: 75.77%
    F-score: 70.79%
    OS: 14.08%
    R-value: 72.44%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 15.81%
    Recall: 68.13%
    F-score: 25.66%
    OS: 330.90%
    R-value: -194.47%
    ---------------------------------------------------------------------------

VQ-CPC DP penalized val:

    # ./eval_segmentation.py vqcpc buckeye val phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 55.72%
    Recall: 74.68%
    F-score: 63.82%
    OS: 34.04%
    R-value: 57.80%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 15.54%
    Recall: 81.03%
    F-score: 26.08%
    OS: 421.37%
    R-value: -266.58%
    ---------------------------------------------------------------------------

TP word segmentation on VQ-VAE DP penalized val:

    # ./eval_segmentation.py vqvae buckeye val wordseg_tp_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 72.27%
    Recall: 28.67%
    F-score: 41.06%
    OS: -60.33%
    R-value: 49.40%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.07%
    Recall: 27.07%
    F-score: 21.67%
    OS: 49.83%
    R-value: 12.43%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 7.49%
    Recall: 10.12%
    F-score: 8.61%
    ---------------------------------------------------------------------------

AG word segmentation on VQ-VAE DP penalized val:

    # ./eval_segmentation.py vqvae buckeye val wordseg_ag_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 69.29%
    Recall: 63.37%
    F-score: 66.20%
    OS: -8.54%
    R-value: 71.27%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 16.49%
    Recall: 56.96%
    F-score: 25.57%
    OS: 245.45%
    R-value: -126.59%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 3.63%
    Recall: 9.92%
    F-score: 5.31%
    ---------------------------------------------------------------------------


Buckeye Felix split
-------------------

VQ-VAE DP penalized test:

    # ./eval_segmentation.py vqvae buckeye_felix test phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 63.80%
    Recall: 77.14%
    F-score: 69.83%
    OS: 20.91%
    R-value: 69.03%
    ---------------------------------------------------------------------------

VQ-VAE DP penalized N-seg. test:

    # ./eval_segmentation.py --phone_tolerance 3 vqvae buckeye_felix test phoneseg_dp_penalized_n_seg
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 61.74%
    Recall: 90.90%
    F-score: 73.53%
    OS: 47.24%
    R-value: 56.03%
    ---------------------------------------------------------------------------
