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

VQ-VAE DP penalized val, dur_weight=4:

    # ./eval_segmentation.py vqvae buckeye val phoneseg_dp_penalized_tune
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 70.54%
    Recall: 71.14%
    F-score: 70.84%
    OS: 0.86%
    R-value: 75.06%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 16.37%
    Recall: 62.36%
    F-score: 25.94%
    OS: 280.91%
    R-value: -154.34%
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

Big CPC DP penalized val, dur_weight=3:

    ./eval_segmentation.py cpc_big buckeye val phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 54.59%
    Recall: 85.91%
    F-score: 66.76%
    OS: 57.36%
    R-value: 45.20%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 14.47%
    Recall: 86.89%
    F-score: 24.81%
    OS: 500.32%
    R-value: -331.77%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 1.87%
    Recall: 8.48%
    F-score: 3.06%
    OS: 354.58%
    ---------------------------------------------------------------------------

Big CPC DP penalized val, dur_weight=5:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 60.18%
    Recall: 79.95%
    F-score: 68.67%
    OS: 32.86%
    R-value: 62.05%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 16.04%
    Recall: 81.27%
    F-score: 26.79%
    OS: 406.79%
    R-value: -254.06%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 2.34%
    Recall: 9.08%
    F-score: 3.72%
    OS: 288.34%
    ---------------------------------------------------------------------------

Big CPC DP penalized val, dur_weight=10:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 67.80%
    Recall: 66.58%
    F-score: 67.19%
    OS: -1.80%
    R-value: 72.09%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 17.79%
    Recall: 66.63%
    F-score: 28.08%
    OS: 274.59%
    R-value: -147.18%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 3.38%
    Recall: 9.95%
    F-score: 5.04%
    OS: 194.71%
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

TP word segmentation on VQ-VAE DP penalized val, Gamma prior, dur_weight=15:

    # ./eval_segmentation.py vqvae buckeye val wordseg_tp_dp_penalized_gamma
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.50%
    Recall: 10.86%
    F-score: 19.08%
    OS: -86.17%
    R-value: 36.96%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 19.15%
    Recall: 10.00%
    F-score: 13.14%
    OS: -47.78%
    R-value: 34.12%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 16.86%
    Recall: 11.16%
    F-score: 13.43%
    OS: -33.76%
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

AG word segmentation on VQ-VAE DP penalized val, Gamma prior, dur_weight=15:

    # ./eval_segmentation.py vqvae buckeye val wordseg_ag_dp_penalized_gamma
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.03%
    Recall: 27.07%
    F-score: 40.19%
    OS: -65.31%
    R-value: 48.35%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 17.29%
    Recall: 22.63%
    F-score: 19.60%
    OS: 30.93%
    R-value: 20.05%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 9.34%
    Recall: 11.39%
    F-score: 10.27%
    OS: 21.86%
    ---------------------------------------------------------------------------


Buckeye Felix split
-------------------

VQ-VAE DP penalized val, dur_weight=3:

    # ./eval_segmentation.py vqvae buckeye_felix val phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 65.64%
    Recall: 75.97%
    F-score: 70.43%
    OS: 15.73%
    R-value: 71.58%
    ---------------------------------------------------------------------------

VQ-VAE DP penalized val, dur_weight=4:

    # ./eval_segmentation.py vqvae buckeye_felix val phoneseg_dp_penalized_tune
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 69.40%
    Recall: 71.23%
    F-score: 70.31%
    OS: 2.64%
    R-value: 74.45%
    ---------------------------------------------------------------------------

VQ-VAE DP penalized val, Poisson prior, dur_weight=2:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 67.46%
    Recall: 70.33%
    F-score: 68.87%
    OS: 4.27%
    R-value: 73.02%
    ---------------------------------------------------------------------------

VQ-VAE DP penalized val, histogram prior, dur_weight=2:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 70.95%
    Recall: 66.54%
    F-score: 68.68%
    OS: -6.22%
    R-value: 73.35%
    ---------------------------------------------------------------------------

VQ-VAE DP penalized val, Gamma prior, dur_weight=2:

    # ./eval_segmentation.py vqvae buckeye_felix val phoneseg_dp_penalized_gamma
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 71.70%
    Recall: 65.85%
    F-score: 68.65%
    OS: -8.17%
    R-value: 73.25%
    ---------------------------------------------------------------------------

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
