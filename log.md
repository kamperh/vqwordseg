2021-02-02
----------

./vq_wordseg.py --algorithm=tp --output_tag wordseg_tp_dp_penalized_gamma vqvae buckeye val phoneseg_dp_penalized_gamma
./eval_segmentation.py vqvae buckeye val wordseg_tp_dp_penalized_gamma

    # relative, ftp
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

    # absolute, ftp
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.08%
    Recall: 20.82%
    F-score: 32.88%
    OS: -73.33%
    R-value: 43.97%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.26%
    Recall: 18.38%
    F-score: 18.32%
    OS: 0.67%
    R-value: 30.09%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 11.34%
    Recall: 11.40%
    F-score: 11.37%
    OS: 0.47%
    ---------------------------------------------------------------------------

    # absolute, mi
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.24%
    Recall: 17.33%
    F-score: 28.38%
    OS: -77.84%
    R-value: 41.52%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.69%
    Recall: 15.63%
    F-score: 17.03%
    OS: -16.37%
    R-value: 32.99%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 13.60%
    Recall: 12.02%
    F-score: 12.76%
    OS: -11.57%
    ---------------------------------------------------------------------------

2021-02-01
----------

**Geometric**

./vq_phoneseg.py --output_tag=phoneseg_dp_penalized_tune --dur_weight=12  vqvae buckeye val
./eval_segmentation.py vqvae buckeye val phoneseg_dp_penalized_tune

./vq_wordseg.py --algorithm=tp --output_tag wordseg_tp_dp_penalized_tune vqvae buckeye val phoneseg_dp_penalized_tune
./eval_segmentation.py vqvae buckeye val wordseg_tp_dp_penalized_tune

    # dur_weight=4
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 73.06%
    Recall: 25.85%
    F-score: 38.19%
    OS: -64.61%
    R-value: 47.45%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.27%
    Recall: 24.42%
    F-score: 20.90%
    OS: 33.64%
    R-value: 20.02%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 8.29%
    Recall: 10.26%
    F-score: 9.17%
    OS: 23.77%
    ---------------------------------------------------------------------------

    # dur_weight=5
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 74.29%
    Recall: 23.97%
    F-score: 36.24%
    OS: -67.74%
    R-value: 46.15%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.45%
    Recall: 22.47%
    F-score: 20.26%
    OS: 21.80%
    R-value: 24.61%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 9.00%
    Recall: 10.39%
    F-score: 9.65%
    OS: 15.41%
    ---------------------------------------------------------------------------

    # dur_weight=6
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 75.30%
    Recall: 22.49%
    F-score: 34.64%
    OS: -70.13%
    R-value: 45.13%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.62%
    Recall: 21.00%
    F-score: 19.74%
    OS: 12.77%
    R-value: 27.54%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 9.67%
    Recall: 10.54%
    F-score: 10.09%
    OS: 9.03%
    ---------------------------------------------------------------------------

    # dur_weight=8
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 77.03%
    Recall: 20.19%
    F-score: 32.00%
    OS: -73.78%
    R-value: 43.53%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.84%
    Recall: 18.64%
    F-score: 18.74%
    OS: -1.04%
    R-value: 30.92%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 10.88%
    Recall: 10.80%
    F-score: 10.84%
    OS: -0.73%
    ---------------------------------------------------------------------------

    # dur_weight=12
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.72%
    Recall: 16.39%
    F-score: 27.13%
    OS: -79.18%
    R-value: 40.86%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.91%
    Recall: 14.86%
    F-score: 16.64%
    OS: -21.42%
    R-value: 33.57%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 13.00%
    Recall: 11.03%
    F-score: 11.93%
    OS: -15.14%
    ---------------------------------------------------------------------------

    # dur_weight=15
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 79.48%
    Recall: 14.22%
    F-score: 24.12%
    OS: -82.11%
    R-value: 39.33%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.94%
    Recall: 12.79%
    F-score: 15.27%
    OS: -32.49%
    R-value: 34.12%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 14.30%
    Recall: 11.02%
    F-score: 12.45%
    OS: -22.96%
    ---------------------------------------------------------------------------

    # dur_weight=20
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 79.51%
    Recall: 11.62%
    F-score: 20.28%
    OS: -85.38%
    R-value: 37.50%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.89%
    Recall: 10.42%
    F-score: 13.43%
    OS: -44.82%
    R-value: 34.09%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 16.00%
    Recall: 10.94%
    F-score: 12.99%
    OS: -31.67%
    ---------------------------------------------------------------------------

    # dur_weight=25
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 79.17%
    Recall: 9.98%
    F-score: 17.73%
    OS: -87.39%
    R-value: 36.34%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.99%
    Recall: 9.04%
    F-score: 12.25%
    OS: -52.41%
    R-value: 33.88%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 17.26%
    Recall: 10.87%
    F-score: 13.34%
    OS: -37.04%
    ---------------------------------------------------------------------------

./vq_wordseg.py --algorithm=ag --output_tag wordseg_ag_dp_penalized_tune vqvae buckeye val phoneseg_dp_penalized_tune
./eval_segmentation.py vqvae buckeye val wordseg_ag_dp_penalized_tune

    # dur_weight=4
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 72.13%
    Recall: 60.04%
    F-score: 65.53%
    OS: -16.76%
    R-value: 70.13%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 16.89%
    Recall: 53.09%
    F-score: 25.62%
    OS: 214.37%
    R-value: -102.10%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 3.97%
    Recall: 9.99%
    F-score: 5.69%
    OS: 151.49%
    ---------------------------------------------------------------------------

**Histogram**

./vq_phoneseg.py --output_tag=phoneseg_dp_penalized_hist --dur_weight_func neg_log_hist --dur_weight 2 vqvae buckeye val
./eval_segmentation.py vqvae buckeye val phoneseg_dp_penalized_hist

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 73.52%
    Recall: 65.33%
    F-score: 69.18%
    OS: -11.13%
    R-value: 73.47%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 17.06%
    Recall: 57.25%
    F-score: 26.28%
    OS: 235.61%
    R-value: -118.15%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 3.92%
    Recall: 10.43%
    F-score: 5.69%
    OS: 166.50%
    ---------------------------------------------------------------------------

./vq_wordseg.py --algorithm=tp --output_tag wordseg_tp_dp_penalized_hist vqvae buckeye val phoneseg_dp_penalized_hist
./eval_segmentation.py vqvae buckeye val wordseg_tp_dp_penalized_hist

    # dur_weight 2
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 72.70%
    Recall: 22.81%
    F-score: 34.72%
    OS: -68.63%
    R-value: 45.33%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.51%
    Recall: 21.93%
    F-score: 20.08%
    OS: 18.48%
    R-value: 25.75%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 9.12%
    Recall: 10.32%
    F-score: 9.68%
    OS: 13.06%
    ---------------------------------------------------------------------------

    # dur_weight 5
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 77.43%
    Recall: 18.34%
    F-score: 29.65%
    OS: -76.32%
    R-value: 42.22%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 19.15%
    Recall: 17.12%
    F-score: 18.08%
    OS: -10.59%
    R-value: 32.67%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 12.10%
    Recall: 11.19%
    F-score: 11.63%
    OS: -7.48%
    ---------------------------------------------------------------------------

    # dur_weight 10
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.63%
    Recall: 14.17%
    F-score: 24.01%
    OS: -81.98%
    R-value: 39.29%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 19.70%
    Recall: 13.40%
    F-score: 15.95%
    OS: -31.98%
    R-value: 34.53%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 14.56%
    Recall: 11.27%
    F-score: 12.71%
    OS: -22.60%
    ---------------------------------------------------------------------------

    # dur_weight 20
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.06%
    Recall: 10.66%
    F-score: 18.76%
    OS: -86.34%
    R-value: 36.82%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 19.14%
    Recall: 9.87%
    F-score: 13.02%
    OS: -48.44%
    R-value: 34.10%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 17.01%
    Recall: 11.19%
    F-score: 13.50%
    OS: -34.23%
    ---------------------------------------------------------------------------

    # dur_weight 25
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 77.77%
    Recall: 9.82%
    F-score: 17.45%
    OS: -87.37%
    R-value: 36.23%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 19.19%
    Recall: 9.15%
    F-score: 12.39%
    OS: -52.32%
    R-value: 33.96%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 17.64%
    Recall: 11.12%
    F-score: 13.64%
    OS: -36.97%
    ---------------------------------------------------------------------------

**Gamma**

./vq_phoneseg.py --output_tag=phoneseg_dp_penalized_gamma --dur_weight_func neg_log_gamma --dur_weight 10 vqvae buckeye val
./eval_segmentation.py vqvae buckeye val phoneseg_dp_penalized_gamma

    # dur_weight 10
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.52%
    Recall: 35.48%
    F-score: 48.88%
    OS: -54.82%
    R-value: 54.24%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 16.88%
    Recall: 28.79%
    F-score: 21.28%
    OS: 70.55%
    R-value: -0.24%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 7.25%
    Recall: 10.86%
    F-score: 8.69%
    OS: 49.86%
    ---------------------------------------------------------------------------

./vq_wordseg.py --algorithm=tp --output_tag wordseg_tp_dp_penalized_gamma vqvae buckeye val phoneseg_dp_penalized_gamma
./eval_segmentation.py vqvae buckeye val wordseg_tp_dp_penalized_gamma

    # dur_weight 10
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.98%
    Recall: 12.95%
    F-score: 22.25%
    OS: -83.60%
    R-value: 38.43%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 19.40%
    Recall: 12.01%
    F-score: 14.83%
    OS: -38.11%
    R-value: 34.42%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 15.40%
    Recall: 11.26%
    F-score: 13.01%
    OS: -26.93%
    ---------------------------------------------------------------------------

    # dur_weight 15
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

    # dur_weight 20
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 77.79%
    Recall: 9.61%
    F-score: 17.10%
    OS: -87.65%
    R-value: 36.08%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 19.21%
    Recall: 8.95%
    F-score: 12.21%
    OS: -53.38%
    R-value: 33.91%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 17.78%
    Recall: 11.07%
    F-score: 13.65%
    OS: -37.73%
    ---------------------------------------------------------------------------

./vq_wordseg.py --algorithm=ag --output_tag wordseg_ag_dp_penalized_gamma vqvae buckeye val phoneseg_dp_penalized_gamma
./eval_segmentation.py vqvae buckeye val wordseg_ag_dp_penalized_gamma

