## 2022-10-04

    ./eval_segmentation.py eskmeans buckeye test eskmeans


## 2021-11-29

    ./vq_wordseg.py --algorithm=tp cpc_big buckeye val phoneseg_merge
    ./eval_segmentation.py cpc_big buckeye val wordseg_tp_merge
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 64.78%
    Recall: 58.90%
    F-score: 61.70%
    OS: -9.07%
    R-value: 67.63%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 49.89%
    Homogeneity: 53.81%
    Completeness: 27.20%
    V-measure: 36.13%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.18%
    Recall: 63.11%
    F-score: 28.23%
    OS: 247.16%
    R-value: -125.38%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 4.40%
    Recall: 12.12%
    F-score: 6.46%
    OS: 175.36%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 13579
    uWER many: 229.36%
    Purity: 21.05%
    Homogeneity: 44.51%
    Completeness: 42.37%
    V-measure: 43.41%
    ---------------------------------------------------------------------------

    ./vq_wordseg.py --algorithm=tp vqvae buckeye val phoneseg_dp_penalized
    ./eval_segmentation.py vqvae buckeye val wordseg_tp_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 72.27%
    Recall: 28.67%
    F-score: 41.06%
    OS: -60.33%
    R-value: 49.40%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 78.41%
    Homogeneity: 84.52%
    Completeness: 28.61%
    V-measure: 42.75%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 19.86%
    Recall: 29.74%
    F-score: 23.81%
    OS: 49.75%
    R-value: 14.52%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 8.15%
    Recall: 11.02%
    F-score: 9.37%
    OS: 35.16%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 50074
    uWER many: 73.70%
    Purity: 70.15%
    Homogeneity: 87.79%
    Completeness: 56.50%
    V-measure: 68.75%
    ---------------------------------------------------------------------------

    ./vq_wordseg.py --algorithm=tp cpc_big buckeye val phoneseg_dp_penalized
    ./eval_segmentation.py cpc_big buckeye val wordseg_tp_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 74.60%
    Recall: 43.70%
    F-score: 55.12%
    OS: -41.41%
    R-value: 59.79%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 53.18%
    Homogeneity: 57.80%
    Completeness: 29.31%
    V-measure: 38.90%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 21.10%
    Recall: 47.16%
    F-score: 29.16%
    OS: 123.46%
    R-value: -29.48%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 7.40%
    Recall: 13.89%
    F-score: 9.66%
    OS: 87.72%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 10619
    uWER many: 148.72%
    Purity: 24.41%
    Homogeneity: 49.41%
    Completeness: 47.59%
    V-measure: 48.48%
    ---------------------------------------------------------------------------

    ./vq_wordseg.py --algorithm=ag cpc_big buckeye val phoneseg_merge
    ./eval_segmentation.py cpc_big buckeye val wordseg_ag_merge
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 66.15%
    Recall: 62.04%
    F-score: 64.03%
    OS: -6.21%
    R-value: 69.54%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 44.61%
    Homogeneity: 47.05%
    Completeness: 25.95%
    V-measure: 33.45%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.21%
    Recall: 65.19%
    F-score: 28.47%
    OS: 258.03%
    R-value: -133.72%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 3.74%
    Recall: 10.59%
    F-score: 5.53%
    OS: 183.06%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 3561
    uWER many: 252.06%
    Purity: 13.77%
    Homogeneity: 36.79%
    Completeness: 37.81%
    V-measure: 37.29%
    ---------------------------------------------------------------------------

    ./vq_wordseg.py --algorithm=dpdp_aernn --dur_weight=3 cpc_big buckeye val phoneseg_merge
    ./eval_segmentation.py cpc_big buckeye val wordseg_dpdp_aernn_merge
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 74.08%
    Recall: 24.25%
    F-score: 36.53%
    OS: -67.27%
    R-value: 46.34%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 84.29%
    Homogeneity: 89.52%
    Completeness: 29.28%
    V-measure: 44.13%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 32.75%
    Recall: 40.87%
    F-score: 36.36%
    OS: 24.78%
    R-value: 38.28%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 20.24%
    Recall: 23.84%
    F-score: 21.90%
    OS: 17.81%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 48661
    uWER many: 51.25%
    Purity: 79.34%
    Homogeneity: 92.13%
    Completeness: 57.72%
    V-measure: 70.98%
    ---------------------------------------------------------------------------

    ./vq_wordseg.py --algorithm=dpdp_aernn --dur_weight=5 cpc_big buckeye val phoneseg_merge
    ./eval_segmentation.py cpc_big buckeye val wordseg_dpdp_aernn_merge
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 76.13%
    Recall: 19.85%
    F-score: 31.49%
    OS: -73.92%
    R-value: 43.28%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 92.19%
    Homogeneity: 95.42%
    Completeness: 30.39%
    V-measure: 46.10%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 35.28%
    Recall: 35.06%
    F-score: 35.17%
    OS: -0.61%
    R-value: 44.79%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 22.46%
    Recall: 22.42%
    F-score: 22.44%
    OS: -0.18%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 48430
    uWER many: 40.07%
    Purity: 89.91%
    Homogeneity: 96.75%
    Completeness: 59.66%
    V-measure: 73.81%
    ---------------------------------------------------------------------------

    ./vq_wordseg.py --algorithm=dpdp_aernn --dur_weight=6 cpc_big buckeye val phoneseg_merge
    ./eval_segmentation.py cpc_big buckeye val wordseg_dpdp_aernn_merge
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 76.85%
    Recall: 18.31%
    F-score: 29.58%
    OS: -76.17%
    R-value: 42.20%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 94.36%
    Homogeneity: 96.83%
    Completeness: 30.71%
    V-measure: 46.63%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 36.02%
    Recall: 32.70%
    F-score: 34.28%
    OS: -9.20%
    R-value: 45.50%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 23.07%
    Recall: 21.62%
    F-score: 22.32%
    OS: -6.26%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 47401
    uWER many: 37.77%
    Purity: 92.81%
    Homogeneity: 97.82%
    Completeness: 60.26%
    V-measure: 74.58%
    ---------------------------------------------------------------------------

    ./vq_wordseg.py --algorithm=dpdp_aernn --dur_weight=3 vqvae buckeye val phoneseg_dp_penalized
    ./eval_segmentation.py vqvae buckeye val wordseg_dpdp_aernn_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 72.97%
    Recall: 32.63%
    F-score: 45.09%
    OS: -55.29%
    R-value: 52.15%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 75.44%
    Homogeneity: 82.08%
    Completeness: 28.07%
    V-measure: 41.83%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 21.97%
    Recall: 37.06%
    F-score: 27.59%
    OS: 68.71%
    R-value: 6.87%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 10.28%
    Recall: 15.28%
    F-score: 12.29%
    OS: 48.56%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 50870
    uWER many: 82.43%
    Purity: 64.61%
    Homogeneity: 84.47%
    Completeness: 55.30%
    V-measure: 66.85%
    ---------------------------------------------------------------------------

    ./vq_wordseg.py --algorithm=dpdp_aernn --dur_weight=5 vqvae buckeye val phoneseg_dp_penalized
    ./eval_segmentation.py vqvae buckeye val wordseg_dpdp_aernn_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 75.10%
    Recall: 21.06%
    F-score: 32.89%
    OS: -71.97%
    R-value: 44.12%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 91.81%
    Homogeneity: 95.13%
    Completeness: 30.52%
    V-measure: 46.21%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 24.40%
    Recall: 25.81%
    F-score: 25.08%
    OS: 5.75%
    R-value: 34.53%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 15.16%
    Recall: 15.78%
    F-score: 15.47%
    OS: 4.06%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 51059
    uWER many: 47.95%
    Purity: 88.57%
    Homogeneity: 96.38%
    Completeness: 59.81%
    V-measure: 73.82%
    ---------------------------------------------------------------------------

    ./vq_wordseg.py --algorithm=dpdp_aernn --dur_weight=6 vqvae buckeye val phoneseg_dp_penalized
    ./eval_segmentation.py vqvae buckeye val wordseg_dpdp_aernn_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 75.46%
    Recall: 16.38%
    F-score: 26.92%
    OS: -78.29%
    R-value: 40.84%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 95.84%
    Homogeneity: 97.74%
    Completeness: 31.24%
    V-measure: 47.35%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 25.72%
    Recall: 21.06%
    F-score: 23.16%
    OS: -18.12%
    R-value: 38.00%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 17.55%
    Recall: 15.31%
    F-score: 16.35%
    OS: -12.80%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 45839
    uWER many: 43.17%
    Purity: 94.28%
    Homogeneity: 98.39%
    Completeness: 61.27%
    V-measure: 75.51%
    ---------------------------------------------------------------------------


## 2021-02-02

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

## 2021-02-01

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

