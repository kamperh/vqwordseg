Phone and Word Segmentation Results
===================================

## Overview

For final results that made it into the paper, search for *final*.


## Buckeye phone segmentation

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

Merged GMM val, MFCCs, 50 components:

    # ./eval_segmentation.py gmm buckeye val phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 29.21%
    Recall: 95.00%
    F-score: 44.69%
    OS: 225.22%
    R-value: -94.03%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 21.61%
    Homogeneity: 19.66%
    Completeness: 17.87%
    V-measure: 18.72%
    ---------------------------------------------------------------------------

DPDP (dur_weight=6), GMM val, MFCCs, 50 components:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 52.96%
    Recall: 83.28%
    F-score: 64.74%
    OS: 57.26%
    R-value: 44.02%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 23.04%
    Homogeneity: 21.63%
    Completeness: 24.04%
    V-measure: 22.77%
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

CPC-big DP penalized val, dur_weight=2:
    
    ./eval_segmentation.py cpc_big buckeye val phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 51.04%
    Recall: 88.75%
    F-score: 64.81%
    OS: 73.87%
    R-value: 32.54%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 36.85%
    Homogeneity: 37.25%
    Completeness: 35.53%
    V-measure: 36.37%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 13.60%
    Recall: 90.20%
    F-score: 23.64%
    OS: 563.17%
    R-value: -384.20%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 1.70%
    Recall: 8.47%
    F-score: 2.83%
    OS: 399.25%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 50
    uWER many: 473.16%
    Purity: 7.44%
    Homogeneity: 15.91%
    Completeness: 28.84%
    V-measure: 20.51%
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

XLSR DP penalized val, dur_weight=2500, *select*:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 49.32%
    Recall: 84.54%
    F-score: 62.30%
    OS: 71.40%
    R-value: 32.76%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 30.37%
    Homogeneity: 31.30%
    Completeness: 28.32%
    V-measure: 29.74%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 13.18%
    Recall: 85.72%
    F-score: 22.85%
    OS: 550.30%
    R-value: -374.85%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 2.33%
    Recall: 11.38%
    F-score: 3.87%
    OS: 388.49%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 50
    uWER many: 463.96%
    Purity: 6.40%
    Homogeneity: 14.15%
    Completeness: 24.51%
    V-measure: 17.94%
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


## Buckeye word segmentation

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

AG word segmentation on MFCC val, *final*:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 56.14%
    Recall: 58.49%
    F-score: 57.29%
    OS: 4.18%
    R-value: 62.98%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 29.48%
    Homogeneity: 35.36%
    Completeness: 16.81%
    V-measure: 22.79%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 16.69%
    Recall: 57.85%
    F-score: 25.90%
    OS: 246.70%
    R-value: -127.26%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 3.10%
    Recall: 9.96%
    F-score: 4.73%
    OS: 221.49%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 7593
    uWER many: 291.89%
    Purity: 11.43%
    Homogeneity: 40.40%
    Completeness: 36.01%
    V-measure: 38.08%
    ---------------------------------------------------------------------------

AG word segmentation on MFCC+GMM test, *final*:
    
    ./eval_segmentation.py gmm buckeye test wordseg_ag_merge
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 54.10%
    Recall: 57.55%
    F-score: 55.77%
    OS: 6.38%
    R-value: 61.27%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 29.49%
    Homogeneity: 35.06%
    Completeness: 16.69%
    V-measure: 22.61%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 15.95%
    Recall: 57.67%
    F-score: 24.99%
    OS: 261.53%
    R-value: -139.89%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 2.83%
    Recall: 9.40%
    F-score: 4.35%
    OS: 231.96%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 8261
    uWER many: 302.12%
    Purity: 11.29%
    Homogeneity: 39.04%
    Completeness: 35.01%
    V-measure: 36.91%
    ---------------------------------------------------------------------------

AG word segmentation on DPDP CPC-big val, *final*:

    ./eval_segmentation.py cpc_big buckeye val wordseg_ag_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 77.41%
    Recall: 52.91%
    F-score: 62.86%
    OS: -31.65%
    R-value: 66.17%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 49.27%
    Homogeneity: 51.99%
    Completeness: 29.83%
    V-measure: 37.91%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 21.72%
    Recall: 56.61%
    F-score: 31.39%
    OS: 160.66%
    R-value: -55.35%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 6.36%
    Recall: 13.61%
    F-score: 8.67%
    OS: 114.08%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 2558
    uWER many: 184.73%
    Purity: 15.43%
    Homogeneity: 39.31%
    Completeness: 42.58%
    V-measure: 40.88%
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

SegAE-RNN Chorowski (dur_weight=4) word segmentation on CPC-big DP penalized
val (dur_weight=5):

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.63%
    Recall: 16.80%
    F-score: 27.69%
    OS: -78.63%
    R-value: 41.15%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 79.61%
    Homogeneity: 86.95%
    Completeness: 29.56%
    V-measure: 44.13%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 35.36%
    Recall: 28.79%
    F-score: 31.74%
    OS: -18.59%
    R-value: 44.60%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 22.99%
    Recall: 20.02%
    F-score: 21.40%
    OS: -12.92%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 30835
    uWER many: 49.58%
    Purity: 74.44%
    Homogeneity: 90.97%
    Completeness: 60.98%
    V-measure: 73.02%
    ---------------------------------------------------------------------------

SegAE-RNN Chorowski (dur_weight=5) word segmentation on CPC-big DP penalized
val (dur_weight=3):

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 76.40%
    Recall: 15.87%
    F-score: 26.28%
    OS: -79.23%
    R-value: 40.48%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 85.20%
    Homogeneity: 90.90%
    Completeness: 29.98%
    V-measure: 45.09%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 36.57%
    Recall: 28.94%
    F-score: 32.31%
    OS: -20.87%
    R-value: 45.22%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 24.15%
    Recall: 20.64%
    F-score: 22.26%
    OS: -14.54%
    ---------------------------------------------------------------------------
    Word clusters:
    uWER many: 44.77%
    Purity: 81.74%
    Homogeneity: 93.95%
    Completeness: 61.28%
    V-measure: 74.18%
    ---------------------------------------------------------------------------

SegAE-RNN Chorowski (dur_weight=4) word segmentation on CPC-big DP penalized
val (dur_weight=3):

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 77.17%
    Recall: 16.63%
    F-score: 27.36%
    OS: -78.45%
    R-value: 41.02%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 83.65%
    Homogeneity: 89.62%
    Completeness: 29.87%
    V-measure: 44.81%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 36.65%
    Recall: 30.09%
    F-score: 33.05%
    OS: -17.90%
    R-value: 45.53%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 24.06%
    Recall: 21.07%
    F-score: 22.47%
    OS: -12.43%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 34671
    uWER many: 45.71%
    Purity: 79.65%
    Homogeneity: 92.85%
    Completeness: 60.79%
    V-measure: 73.48%
    ---------------------------------------------------------------------------

SegAE-RNN Chorowski (dur_weight=4) word segmentation on CPC-big DP penalized
val (dur_weight=2), training on first 10k for 5 epochs, *best*:

     ./eval_segmentation.py cpc_big buckeye val wordseg_segaernn_dp_penalized_tune
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.80%
    Recall: 18.62%
    F-score: 30.12%
    OS: -76.37%
    R-value: 42.43%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 83.07%
    Homogeneity: 89.31%
    Completeness: 29.67%
    V-measure: 44.54%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 36.78%
    Recall: 33.12%
    F-score: 34.85%
    OS: -9.96%
    R-value: 46.07%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 24.64%
    Recall: 22.96%
    F-score: 23.77%
    OS: -6.80%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 36308
    uWER many: 45.40%
    Purity: 78.45%
    Homogeneity: 92.44%
    Completeness: 60.16%
    V-measure: 72.89%
    ---------------------------------------------------------------------------

SegAE-RNN Chorowski (dur_weight=2) word segmentation on CPC-big DP penalized
val (dur_weight=2), training on all for 5 epochs:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 75.99%
    Recall: 19.96%
    F-score: 31.61%
    OS: -73.74%
    R-value: 43.36%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 78.17%
    Homogeneity: 85.46%
    Completeness: 29.05%
    V-measure: 43.36%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 33.80%
    Recall: 33.83%
    F-score: 33.81%
    OS: 0.09%
    R-value: 43.49%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 22.05%
    Recall: 22.12%
    F-score: 22.08%
    OS: 0.31%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 34608
    uWER many: 51.43%
    Purity: 71.65%
    Homogeneity: 89.23%
    Completeness: 59.25%
    V-measure: 71.22%
    ---------------------------------------------------------------------------

SegAE-RNN Chorowski (dur_weight=3) word segmentation on CPC-big DP penalized
val (dur_weight=2), *final*:

    ./eval_segmentation.py cpc_big buckeye val wordseg_dpdp_aernn_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.32%
    Recall: 20.84%
    F-score: 32.92%
    OS: -73.39%
    R-value: 43.99%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 77.42%
    Homogeneity: 84.85%
    Completeness: 29.00%
    V-measure: 43.23%
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
    Word clusters:
    No. clusters: 33336
    uWER many: 51.15%
    Purity: 70.36%
    Homogeneity: 88.59%
    Completeness: 59.09%
    V-measure: 70.90%
    ---------------------------------------------------------------------------

Merged CPC-big test, *final*:

    ./eval_segmentation.py cpc_big buckeye test phoneseg_merge
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 35.55%
    Recall: 93.94%
    F-score: 51.59%
    OS: 164.21%
    R-value: -42.36%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 32.33%
    Homogeneity: 31.21%
    Completeness: 29.88%
    V-measure: 30.53%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 9.12%
    Recall: 94.49%
    F-score: 16.63%
    OS: 936.59%
    R-value: -701.39%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 0.86%
    Recall: 6.47%
    F-score: 1.52%
    OS: 650.28%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 50
    uWER many: 721.99%
    Purity: 6.76%
    Homogeneity: 12.55%
    Completeness: 22.73%
    V-measure: 16.17%
    ---------------------------------------------------------------------------


SegAE-RNN Chorowski (dur_weight=3) word segmentation on CPC-big DP penalized
test (dur_weight=2), *final*:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 77.86%
    Recall: 21.16%
    F-score: 33.27%
    OS: -72.82%
    R-value: 44.21%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 75.15%
    Homogeneity: 82.81%
    Completeness: 28.25%
    V-measure: 42.13%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 35.30%
    Recall: 37.66%
    F-score: 36.44%
    OS: 6.68%
    R-value: 44.25%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 24.40%
    Recall: 25.59%
    F-score: 24.98%
    OS: 4.88%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 38981
    uWER: 105.00%
    uWER many: 54.13%
    Purity: 67.10%
    Homogeneity: 86.92%
    Completeness: 58.14%
    V-measure: 69.68%
    ---------------------------------------------------------------------------

SegAE-RNN Chorowski (dur_weight=3) word segmentation on CPC-big DP penalized
val (dur_weight=1):

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 75.73%
    Recall: 20.71%
    F-score: 32.52%
    OS: -72.65%
    R-value: 43.88%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 80.85%
    Homogeneity: 87.22%
    Completeness: 29.12%
    V-measure: 43.66%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 34.94%
    Recall: 36.41%
    F-score: 35.66%
    OS: 4.22%
    R-value: 44.16%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 23.21%
    Recall: 23.96%
    F-score: 23.58%
    OS: 3.25%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 38268
    uWER many: 48.26%
    Purity: 75.26%
    Homogeneity: 90.70%
    Completeness: 58.93%
    V-measure: 71.44%
    ---------------------------------------------------------------------------

SegAE-RNN Chorowski (dur_weight=4) word segmentation on CPC-big DP penalized
val (dur_weight=0.1):

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 75.17%
    Recall: 18.91%
    F-score: 30.21%
    OS: -74.85%
    R-value: 42.61%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 91.13%
    Homogeneity: 94.76%
    Completeness: 30.32%
    V-measure: 45.94%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 34.42%
    Recall: 33.00%
    F-score: 33.70%
    OS: -4.14%
    R-value: 44.21%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 22.46%
    Recall: 21.86%
    F-score: 22.15%
    OS: -2.68%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 46416
    uWER many: 40.52%
    Purity: 88.69%
    Homogeneity: 96.36%
    Completeness: 59.84%
    V-measure: 73.83%
    ---------------------------------------------------------------------------

SegAE-RNN Chorowski (dur_weight=3) word segmentation on XLSR DPDP val
(dur_weight=2500):

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 72.49%
    Recall: 16.27%
    F-score: 26.57%
    OS: -77.56%
    R-value: 40.75%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 85.83%
    Homogeneity: 90.63%
    Completeness: 30.04%
    V-measure: 45.12%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 26.81%
    Recall: 22.80%
    F-score: 24.64%
    OS: -14.94%
    R-value: 38.67%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 17.59%
    Recall: 15.74%
    F-score: 16.61%
    OS: -10.55%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 38852
    uWER many: 49.85%
    Purity: 81.68%
    Homogeneity: 93.28%
    Completeness: 60.98%
    V-measure: 73.75%
    ---------------------------------------------------------------------------

TP, DPDP GMM val, MFCCs, 50 components:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 59.17%
    Recall: 60.21%
    F-score: 59.69%
    OS: 1.76%
    R-value: 65.40%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 42.35%
    Homogeneity: 50.11%
    Completeness: 20.96%
    V-measure: 29.56%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 17.94%
    Recall: 60.73%
    F-score: 27.69%
    OS: 238.59%
    R-value: -119.14%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 3.80%
    Recall: 11.96%
    F-score: 5.77%
    OS: 214.65%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 35842
    uWER many: 256.14%
    Purity: 26.65%
    Homogeneity: 54.32%
    Completeness: 42.78%
    V-measure: 47.86%
    ---------------------------------------------------------------------------

DPDP AE-RNN (dur_weight=3), DPDP GMM val, MFCCs, 50 components:

    ./eval_segmentation.py gmm buckeye val wordseg_dpdp_aernn_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 65.03%
    Recall: 29.10%
    F-score: 40.21%
    OS: -55.25%
    R-value: 49.52%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 86.63%
    Homogeneity: 91.64%
    Completeness: 28.96%
    V-measure: 44.02%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 23.63%
    Recall: 35.15%
    F-score: 28.26%
    OS: 48.78%
    R-value: 19.25%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 11.76%
    Recall: 18.18%
    F-score: 14.28%
    OS: 54.59%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 70916
    uWER many: 74.95%
    Purity: 83.19%
    Homogeneity: 94.21%
    Completeness: 57.22%
    V-measure: 71.20%
    ---------------------------------------------------------------------------

DPDP AE-RNN (dur_weight=6), DPDP GMM val, MFCCs, 50 components, *final*:

    ./eval_segmentation.py gmm buckeye val wordseg_dpdp_aernn_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 67.76%
    Recall: 20.62%
    F-score: 31.62%
    OS: -69.58%
    R-value: 43.75%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 97.15%
    Homogeneity: 98.54%
    Completeness: 30.56%
    V-measure: 46.65%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 26.45%
    Recall: 26.74%
    F-score: 26.60%
    OS: 1.11%
    R-value: 37.07%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 16.15%
    Recall: 18.48%
    F-score: 17.24%
    OS: 14.39%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 61639
    uWER many: 43.75%
    Purity: 96.45%
    Homogeneity: 99.05%
    Completeness: 59.89%
    V-measure: 74.64%
    ---------------------------------------------------------------------------

DPDP AE-RNN (dur_weight=3), DPDP GMM val, MFCCs, 25 components:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 65.83%
    Recall: 25.94%
    F-score: 37.22%
    OS: -60.60%
    R-value: 47.39%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 77.07%
    Homogeneity: 84.62%
    Completeness: 27.75%
    V-measure: 41.79%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 23.89%
    Recall: 31.29%
    F-score: 27.09%
    OS: 30.99%
    R-value: 27.06%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 12.45%
    Recall: 17.38%
    F-score: 14.51%
    OS: 39.58%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 54535
    uWER many: 71.47%
    Purity: 72.20%
    Homogeneity: 89.50%
    Completeness: 56.52%
    V-measure: 69.28%
    ---------------------------------------------------------------------------

DPDP AE-RNN (dur_weight=6), DPDP GMM val, MFCCs, 25 components:

    ./eval_segmentation.py gmm buckeye val wordseg_segaernn_dp_penalized_25
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 69.00%
    Recall: 17.68%
    F-score: 28.15%
    OS: -74.38%
    R-value: 41.72%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 94.27%
    Homogeneity: 96.84%
    Completeness: 30.45%
    V-measure: 46.33%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 26.85%
    Recall: 22.86%
    F-score: 24.69%
    OS: -14.86%
    R-value: 38.70%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 16.70%
    Recall: 16.85%
    F-score: 16.77%
    OS: 0.92%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 52243
    uWER many: 42.06%
    Purity: 93.20%
    Homogeneity: 98.05%
    Completeness: 60.43%
    V-measure: 74.77%
    ---------------------------------------------------------------------------


## Buckeye lexicon building

K-means clustering on the AE-RNN embeddings:

    ./eval_segmentation.py cpc_big buckeye val wordseg_dpdp_aernn_dp_penalized_kmeans14000    
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 78.53%
    Recall: 20.24%
    F-score: 32.19%
    OS: -74.22%
    R-value: 43.57%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 54.81%
    Homogeneity: 70.66%
    Completeness: 26.04%
    V-measure: 38.05%
    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 36.31%
    Recall: 35.67%
    F-score: 35.99%
    OS: -1.78%
    R-value: 45.70%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 24.09%
    Recall: 23.84%
    F-score: 23.97%
    OS: -1.01%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 14000
    uWER many: 72.18%
    Purity: 42.74%
    Homogeneity: 79.72%
    Completeness: 57.32%
    V-measure: 66.69%
    ---------------------------------------------------------------------------


## Xitsonga

SegAE-RNN Chorowski (dur_weight=3) word segmentation on CPC-big DP penalized
val (dur_weight=2):

    ---------------------------------------------------------------------------
    Word boundaries:
    Precision: 20.59%
    Recall: 19.75%
    F-score: 20.16%
    OS: -4.06%
    R-value: 32.89%
    ---------------------------------------------------------------------------
    Word token boundaries:
    Precision: 6.34%
    Recall: 6.14%
    F-score: 6.24%
    OS: -3.20%
    ---------------------------------------------------------------------------
    Word clusters:
    No. clusters: 15718
    uWER many: 51.12%
    Purity: 84.67%
    Homogeneity: 93.22%
    Completeness: 65.88%
    V-measure: 77.20%
    ---------------------------------------------------------------------------


## Buckeye Felix split

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

DPDP VQ-CPC val, dur_weight=700:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 70.55%
    Recall: 69.92%
    F-score: 70.23%
    OS: -0.89%
    R-value: 74.64%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 38.48%
    Homogeneity: 46.17%
    Completeness: 27.05%
    V-measure: 34.12%
    ---------------------------------------------------------------------------

DPDP VQ-CPC val, dur_weight=800:

    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 71.93%
    Recall: 65.58%
    F-score: 68.61%
    OS: -8.83%
    R-value: 73.18%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 38.70%
    Homogeneity: 46.61%
    Completeness: 27.34%
    V-measure: 34.46%
    ---------------------------------------------------------------------------

DPDP VQ-CPC test, dur_weight=700, *final*:
    
    ./eval_segmentation.py --phone_tolerance 3 vqcpc buckeye_felix test phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 69.64%
    Recall: 73.41%
    F-score: 71.48%
    OS: 5.42%
    R-value: 75.12%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 38.83%
    Homogeneity: 45.94%
    Completeness: 27.01%
    V-measure: 34.02%
    ---------------------------------------------------------------------------

DPDP CPC-big val, dur_weight=10:

    ./eval_segmentation.py cpc_big buckeye_felix val phoneseg_dp_penalized_tune
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 65.46%
    Recall: 65.78%
    F-score: 65.62%
    OS: 0.48%
    R-value: 70.62%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 44.36%
    Homogeneity: 47.30%
    Completeness: 44.13%
    V-measure: 45.66%
    ---------------------------------------------------------------------------

DPDP CPC-big test, dur_weight=10, *final*:

    ./eval_segmentation.py --phone_tolerance 3 cpc_big buckeye_felix test phoneseg_dp_penalized
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 73.15%
    Recall: 77.68%
    F-score: 75.35%
    OS: 6.19%
    R-value: 78.34%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 45.55%
    Homogeneity: 48.81%
    Completeness: 45.62%
    V-measure: 47.16%
    ---------------------------------------------------------------------------

Merged CPC-big test, *final*:
    
    ./eval_segmentation.py --phone_tolerance 3 cpc_big buckeye_felix test phoneseg_merge
    ---------------------------------------------------------------------------
    Phone boundaries:
    Precision: 36.89%
    Recall: 97.20%
    F-score: 53.48%
    OS: 163.47%
    R-value: -40.54%
    ---------------------------------------------------------------------------
    Phone clusters:
    Purity: 35.73%
    Homogeneity: 38.38%
    Completeness: 36.51%
    V-measure: 37.42%
    ---------------------------------------------------------------------------


## ZeroSpeech'17 Mandarin

DP penalized CPC-big (dur_weight=2), DP penalized AE-RNN (dur_weight=3),
*final*:

    boundary_precision: 0.6616161616161617
    boundary_recall: 0.7065405514187966
    boundary_fscore: 0.6833407945189482
    token_precision: 0.2688495201940314
    type_precision: 0.25486822554156563
    token_recall: 0.257577288341079
    type_recall: 0.3673768459023785
    token_fscore: 0.2630927196739074
    type_fscore: 0.30095114969064546
    words: 12787
    coverage: 0.9999693444306494
    ned: 0.8897823993837859
    pairs: 17310

DP penalized CPC-big (dur_weight=2), DP penalized AE-RNN (dur_weight=1):

    boundary_precision: 0.5922140060896042
    boundary_recall: 0.7671966190380358
    boundary_fscore: 0.6684434781083972
    token_precision: 0.23382054243498346
    type_precision: 0.26392887383573244
    token_recall: 0.29657506566983227
    type_recall: 0.35136963138315863
    token_fscore: 0.2614853579779534
    type_fscore: 0.3014361007688216
    words: 11810
    coverage: 0.9998313943685719
    ned: 0.958675171906796
    pairs: 564190

DP penalized CPC-big (dur_weight=1), DP penalized AE-RNN (dur_weight=3):

    boundary_precision: 0.6525670837485698
    boundary_recall: 0.7116119943650634
    boundary_fscore: 0.6808117370711232
    token_precision: 0.2663520576444918
    type_precision: 0.25612908279157204
    token_recall: 0.265154576682158
    type_recall: 0.36861684139330403
    token_fscore: 0.26575196820494645
    type_fscore: 0.3022460486181717
    words: 12767
    coverage: 0.9999693444306494
    ned: 0.9174382816394651
    pairs: 15379

DP penalized CPC-big (dur_weight=1), DP penalized AE-RNN (dur_weight=1):

    boundary_precision: 0.573416549705856
    boundary_recall: 0.7728718051921916
    boundary_fscore: 0.6583693341562092
    token_precision: 0.21634615384615385
    type_precision: 0.25046713096653644
    token_recall: 0.28869468579511015
    type_recall: 0.33243151843084207
    token_fscore: 0.24733835367437032
    type_fscore: 0.2856866069266166
    words: 11774
    coverage: 0.9997700832298708
    ned: 0.9616700675645037
    pairs: 477815

DP penalized XLSR (dur_weight=2500), DP penalized AE-RNN (dur_weight=3):

    boundary_precision: 0.5780879175374588
    boundary_recall: 0.6568726101831355
    boundary_fscore: 0.6149672168211621
    token_precision: 0.1839140413102441
    type_precision: 0.17905038057267125
    token_recall: 0.17811679127096383
    type_recall: 0.27843535114417767
    token_fscore: 0.18096900020529666
    type_fscore: 0.21794758669372627
    words: 13795
    coverage: 0.9997394276605203
    ned: 0.8476354658834546
    pairs: 7831

DP penalized XLSR (dur_weight=1500), DP penalized AE-RNN (dur_weight=3):

    boundary_precision: 0.5632707950669116
    boundary_recall: 0.6912054739384182
    boundary_fscore: 0.6207145826179676
    token_precision: 0.19804022692109335
    type_precision: 0.1994954954954955
    token_recall: 0.2133764396847848
    type_recall: 0.3120279562619772
    token_fscore: 0.2054224924012158
    type_fscore: 0.24338345203552275
    words: 13875
    coverage: 0.9997394276605203
    ned: 0.8672395157803717
    pairs: 15420

DP penalized XLSR (dur_weight=500), DP penalized AE-RNN (dur_weight=3):

    boundary_precision: 0.5272764681676563
    boundary_recall: 0.7220366270879452
    boundary_fscore: 0.6094755975334228
    token_precision: 0.18828590337524817
    type_precision: 0.1983859649122807
    token_recall: 0.22994544352394422
    type_recall: 0.3186788411678503
    token_fscore: 0.20704084417356497
    type_fscore: 0.24453959603823366
    words: 14250
    coverage: 0.999724099875845
    ned: 0.8283892604551045
    pairs: 17344


## ZeroSpeech'17 French

DP penalized CPC-big (dur_weight=2), DP penalized AE-RNN (dur_weight=3),
segment length<=50:

    boundary_precision: 0.4638637792059966
    boundary_recall: 0.5817915641948714
    boundary_fscore: 0.5161777929616368
    token_precision: 0.10097769204095705
    type_precision: 0.04817722938867078
    token_recall: 0.11336990763943582
    type_recall: 0.15810058436479088
    token_fscore: 0.10681558027611417
    type_fscore: 0.07385038633896811
    words: 71320
    coverage: 0.9994668206384628
    ned: 0.7985821903307203
    pairs: 5642231

DP penalized CPC-big (dur_weight=2), DP penalized AE-RNN (dur_weight=3),
*final*:

    boundary_precision: 0.4981988292390054
    boundary_recall: 0.5787582779652644
    boundary_fscore: 0.5354655158796178
    token_precision: 0.11984520550303907
    type_precision: 0.0522429243423646
    token_recall: 0.1245838404811341
    type_recall: 0.18294759122072424
    token_fscore: 0.12216859010926324
    type_fscore: 0.08127638262860414
    words: 76106
    coverage: 0.9994785245756672
    ned: 0.8678625502797006
    pairs: 4482962

DP penalized CPC-big (dur_weight=2), DP penalized AE-RNN (dur_weight=1):

    boundary_precision: 0.44045020257251954
    boundary_recall: 0.6867723291642297
    boundary_fscore: 0.5366979509104225
    token_precision: 0.11128960259377166
    type_precision: 0.05380097401918776
    token_recall: 0.16957381685401304
    type_recall: 0.11843739934661575
    token_fscore: 0.1343841980212064
    type_fscore: 0.07399103139013453
    words: 47843
    coverage: 0.9992093340199644
    ned: 0.7407467270862337
    pairs: 34973886

DP penalized CPC-big (dur_weight=1), DP penalized AE-RNN (dur_weight=1):

    boundary_precision: 0.43373805566862395
    boundary_recall: 0.7006309961893207
    boundary_fscore: 0.5357874062776931
    token_precision: 0.10650271854604094
    type_precision: 0.05249721804019114
    token_recall: 0.17495704159805256
    type_recall: 0.11070721943588092
    token_fscore: 0.13240543194825424
    type_fscore: 0.07122136048783377
    words: 45831
    coverage: 0.998635841096945
    ned: 0.7472744666179789
    pairs: 46830800

DP penalized XLSR (dur_weight=1500), DP penalized AE-RNN (dur_weight=1):

    boundary_precision: 0.434751257850677
    boundary_recall: 0.5891731661332684
    boundary_fscore: 0.5003177364823178
    token_precision: 0.08654744268318969
    type_precision: 0.04939499419857451
    token_recall: 0.11319538913152431
    type_recall: 0.1508305342106474
    token_fscore: 0.09809384761251069
    type_fscore: 0.07441881583726843
    words: 66363
    coverage: 0.9979388066145451
    ned: 0.5986752611793141
    pairs: 10054784


## ZeroSpeech'17 English

DP penalized CPC-big (dur_weight=2), DP penalized AE-RNN (dur_weight=3),
*final*:

    boundary_precision: 0.5272349929534665
    boundary_recall: 0.6328394074696406
    boundary_fscore: 0.5752304859347639
    token_precision: 0.18578293430444576
    type_precision: 0.06063438714420754
    token_recall: 0.20514162595681987
    type_recall: 0.2744864967668315
    token_fscore: 0.19498295626538212
    type_fscore: 0.09932726553225167
    words: 95210
    coverage: 0.9988127448148101
    ned: 0.5709890053444755
    pairs: 10327401


## ZeroSpeech'17 LANG1 (German)

These were generated by Robin and can be found in
`vqseg/zerospeech2017_eval/kamper_submission.zip`. 

    metric: boundary
    precision: 0.5053194153293602
    recall: 0.6174457334184231
    fscore: 0.5557837582625119

    metric: token
    precision: 0.14448148897050178
    recall: 0.16002854211263717
    fscore: 0.15185813064988854
    metric: type
    precision: 0.05823323136611207
    recall: 0.19390501428512277
    fscore: 0.08956765389360137


## ZeroSpeech'17 LANG2 (Wolof)

    metric: boundary
    precision: 0.631188814763694
    recall: 0.5646536940068141
    fscore: 0.5960702906246983

    metric: token
    precision: 0.17489360154232295
    recall: 0.1308940433409561
    fscore: 0.14972829048783146
    metric: type
    precision: 0.08042438928548039
    recall: 0.4967637540453074
    fscore: 0.1384363902823967
