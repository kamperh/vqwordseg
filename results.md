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

CPC-big normalized DP penalized val, dur_weight=200:

    ./eval_segmentation.py cpc_big_normalized buckeye val phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 55.42%
    Recall: 85.47%
    F-score: 67.24%
    OS: 54.23%
    R-value: 47.62%
    ---------------------------------------------------------------------------
    Clusters:
    Phone purity: 36.86%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 14.91%
    Recall: 87.52%
    F-score: 25.48%
    OS: 486.97%
    R-value: -320.15%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 1.90%
    Recall: 8.47%
    F-score: 3.11%
    OS: 345.42%
    ---------------------------------------------------------------------------

CPC-big DP penalized val, Gamma:

    # ./utils/eval_segmentation.py cpc_big buckeye val phoneseg_dp_penalized_gamma
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 62.81%
    Recall: 76.87%
    F-score: 69.13%
    OS: 22.39%
    R-value: 67.81%
    ---------------------------------------------------------------------------
    Clusters:
    Phone purity: 40.07%
    Phone homogeneity: 40.95%
    Phone completeness: 38.84%
    Phone V-measure: 39.87%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 17.09%
    Recall: 79.75%
    F-score: 28.14%
    OS: 366.72%
    R-value: -220.46%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 2.73%
    Recall: 9.83%
    F-score: 4.27%
    OS: 260.07%
    ---------------------------------------------------------------------------    

CPC-big DP penalized val, dur_weight=3:

    ./eval_segmentation.py cpc_big buckeye val phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 54.59%
    Recall: 85.91%
    F-score: 66.76%
    OS: 57.36%
    R-value: 45.20%
    ---------------------------------------------------------------------------
    Clusters:
    Phone purity: 38.10%
    Phone homogeneity: 38.65%
    Phone completeness: 36.77%
    Phone V-measure: 37.69%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 14.62%
    Recall: 87.73%
    F-score: 25.06%
    OS: 500.14%
    R-value: -331.31%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 1.94%
    Recall: 8.82%
    F-score: 3.18%
    OS: 354.59%
    ---------------------------------------------------------------------------

CPC-big DP penalized val, dur_weight=5:

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

CPC-big DP penalized val, dur_weight=10:

    ./eval_segmentation.py cpc_big buckeye val phoneseg_dp_penalized_tune
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 67.80%
    Recall: 66.58%
    F-score: 67.19%
    OS: -1.80%
    R-value: 72.09%
    ---------------------------------------------------------------------------
    Clusters:
    Phone purity: 41.74%
    Phone homogeneity: 42.57%
    Phone completeness: 40.42%
    Phone V-measure: 41.47%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.04%
    Recall: 67.56%
    F-score: 28.48%
    OS: 274.48%
    R-value: -146.70%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 3.57%
    Recall: 10.51%
    F-score: 5.33%
    OS: 194.71%
    ---------------------------------------------------------------------------

CPC-big DP penalized val, dur_weight=50:

    ./utils/eval_segmentation.py cpc_big buckeye val dp_penalized_tune
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 69.48%
    Recall: 36.79%
    F-score: 48.11%
    OS: -47.04%
    R-value: 54.89%
    ---------------------------------------------------------------------------
    Clusters:
    Phone purity: 40.20%
    Phone homogeneity: 38.64%
    Phone completeness: 37.47%
    Phone V-measure: 38.05%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.75%
    Recall: 37.88%
    F-score: 25.08%
    OS: 102.04%
    R-value: -17.77%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 6.88%
    Recall: 11.86%
    F-score: 8.71%
    OS: 72.54%
    ---------------------------------------------------------------------------

ResDAVENet-VQ merged val:

    ./eval_segmentation.py resdavenet_vq buckeye val phoneseg_merge
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 30.99%
    Recall: 98.06%
    F-score: 47.10%
    OS: 216.40%
    R-value: -85.40%
    ---------------------------------------------------------------------------
    Clusters:
    Phone purity: 34.65%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 9.79%
    Recall: 95.72%
    F-score: 17.77%
    OS: 877.57%
    R-value: -650.57%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 0.64%
    Recall: 5.95%
    F-score: 1.16%
    OS: 829.73%
    ---------------------------------------------------------------------------

ResDAVENet-VQ DP penalized val, dur_weight=50:

    ./eval_segmentation.py resdavenet_vq buckeye val phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 45.52%
    Recall: 86.07%
    F-score: 59.55%
    OS: 89.07%
    R-value: 18.51%
    ---------------------------------------------------------------------------
    Clusters:
    Phone purity: 35.50%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 13.67%
    Recall: 79.84%
    F-score: 23.35%
    OS: 483.95%
    R-value: -320.41%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 1.55%
    Recall: 8.77%
    F-score: 2.63%
    OS: 467.21%
    ---------------------------------------------------------------------------

ResDAVENet-VQ DP penalized val, dur_weight=200:

    ./eval_segmentation.py resdavenet_vq buckeye val phoneseg_dp_penalized_tune
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 59.44%
    Recall: 58.29%
    F-score: 58.86%
    OS: -1.94%
    R-value: 65.06%
    ---------------------------------------------------------------------------
    Clusters:
    Phone purity: 39.61%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 18.53%
    Recall: 56.11%
    F-score: 27.86%
    OS: 202.78%
    R-value: -90.95%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 3.44%
    Recall: 10.61%
    F-score: 5.20%
    OS: 208.26%
    ---------------------------------------------------------------------------

ResDAVENet-VQ clustered codebook DP penalized val, dur_weight=50:

    ./eval_segmentation.py resdavenet_vq_clust50 buckeye val phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 51.07%
    Recall: 76.11%
    F-score: 61.13%
    OS: 49.02%
    R-value: 46.95%
    ---------------------------------------------------------------------------
    Clusters:
    Phone purity: 27.63%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 15.64%
    Recall: 71.97%
    F-score: 25.69%
    OS: 360.22%
    R-value: -217.92%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 2.09%
    Recall: 9.48%
    F-score: 3.43%
    OS: 353.26%
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

AG word segmentation on CPC-big DP penalized val, dur_weight=3:
    
    # ./utils/eval_segmentation.py cpc_big buckeye val wordseg_ag_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.76%
    Recall: 51.77%
    F-score: 62.48%
    OS: -34.27%
    R-value: 65.48%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 22.07%
    Recall: 55.32%
    F-score: 31.55%
    OS: 150.63%
    R-value: -47.61%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 6.85%
    Recall: 14.17%
    F-score: 9.23%
    OS: 106.97%
    ---------------------------------------------------------------------------

SegAE-RNN Gamma word segmentation on CPC-big DP penalized val, dur_weight=3:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 76.82%
    Recall: 23.03%
    F-score: 35.44%
    OS: -70.02%
    R-value: 45.52%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 69.88%
    Homogeneity: 78.74%
    Completeness: 28.07%
    V-measure: 41.39%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 33.75%
    Recall: 38.55%
    F-score: 35.99%
    OS: 14.22%
    R-value: 41.71%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 22.14%
    Recall: 24.43%
    F-score: 23.23%
    OS: 10.33%
    ---------------------------------------------------------------------------
    Word clusters:
    Purity: 59.36%
    Homogeneity: 83.10%
    Completeness: 57.88%
    V-measure: 68.24%
    ---------------------------------------------------------------------------

SegAE-RNN Chorowski (dur_weight = 5) word segmentation on CPC-big DP penalized
val (dur_weight=3):

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.83%
    Recall: 16.53%
    F-score: 27.32%
    OS: -79.03%
    R-value: 40.95%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 85.20%
    Homogeneity: 90.98%
    Completeness: 30.12%
    V-measure: 45.25%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 36.81%
    Recall: 29.40%
    F-score: 32.69%
    OS: -20.13%
    R-value: 45.45%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 23.96%
    Recall: 20.60%
    F-score: 22.16%
    OS: -14.01%
    ---------------------------------------------------------------------------
    Word clusters:
    Purity: 81.49%
    Homogeneity: 93.94%
    Completeness: 61.29%
    V-measure: 74.18%
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
