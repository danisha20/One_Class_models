# One-Class-Classifiers Evaluation

Anomaly Detection consists of detecting data points that reside outside the range of the majority class in a data set. The technique falls under the category of unsupervised learning as the algorithms are trained without any label. 

In this project, we fit thirty different one-class classifiers with unlabeled examples from 90 different datasets from the KEEL-dataset repository [1]. We assess the results of these classifiers by using the true labels and compare their Area Under the Curve (AUC) and Average Precision performances. To provide a comparison between the performance of the algorithms, we provide an evaluation with hypothesis testing using the Friedman test and the post-hoc pairwise Nemenyi test [2]. 

## Objective: 

To provide an evaluation of different one-class classifiers for the task of anomaly detection. The objective is to demonstrate which of them has a better performance by analyzing their ranking results in multiple datasets. 

## Datasets: 

95 KEEL datasets[1]

| 1  | abalone-17_vs_7-8-9-10    | 24 | ecoli-0-6-7_vs_3-5     | 48 | kr-vs-k-three_vs_eleven        | 72 | vowel0                     |   |
|----|---------------------------|----|------------------------|----|--------------------------------|----|----------------------------|---|
| 2  | abalone-19_vs_10-11-12-13 | 25 | ecoli-0-6-7_vs_5       | 49 | kr-vs-k-zero-one_vs_draw       | 73 | winequality-red-3_vs_5     |   |
| 3  | abalone-20_vs_8-9-10      | 26 | ecoli-0_vs_1           | 50 | kr-vs-k-zero_vs_eight          | 74 | winequality-red-4          |   |
| 4  | abalone-21_vs_8           | 27 | ecoli1                 | 51 | kr-vs-k-zero_vs_fifteen        | 75 | winequality-red-8_vs_6-7   |   |
| 5  | abalone-3_vs_11           | 28 | ecoli2                 | 52 | led7digit-0-2-4-5-6-7-8-9_vs_1 | 76 | winequality-red-8_vs_6     |   |
| 6  | abalone19                 | 29 | ecoli3                 | 53 | lymphography-normal-fibrosis   | 77 | winequality-white-3-9_vs_5 |   |
| 7  | abalone9-18               | 30 | ecoli4                 | 54 | new-thyroid1                   | 78 | winequality-white-3_vs_7   |   |
| 8  | car-good                  | 31 | flare-F                | 55 | new-thyroid2                   | 79 | winequality-white-9_vs_4   |   |
| 9  | car-vgood                 | 32 | glass-0-1-2-3_vs_4-5-6 | 56 | page-blocks-1-3_vs_4           | 80 | wisconsinImb               |   |
| 10 | cleveland-0_vs_4          | 33 | glass-0-1-4-6_vs_2     | 57 | page-blocks0                   | 81 | yeast-0-2-5-6_vs_3-7-8-9   |   |
| 11 | dermatology-6             | 34 | glass-0-1-5_vs_2       | 58 | pimaImb                        | 82 | yeast-0-2-5-7-9_vs_3-6-8   |   |
| 12 | ecoli-0-1-3-7_vs_2-6      | 35 | glass-0-1-6_vs_2       | 59 | poker-8-9_vs_5                 | 83 | yeast-0-3-5-9_vs_7-8       |   |
| 13 | ecoli-0-1-4-6_vs_5        | 36 | glass-0-1-6_vs_5       | 60 | poker-8-9_vs_6                 | 84 | yeast-0-5-6-7-9_vs_4       |   |
| 14 | ecoli-0-1-4-7_vs_2-3-5-6  | 37 | glass-0-4_vs_5         | 61 | poker-8_vs_6                   | 85 | yeast-1-2-8-9_vs_7         |   |
| 15 | ecoli-0-1-4-7_vs_5-6      | 38 | glass-0-6_vs_5         | 62 | poker-9_vs_7                   | 86 | yeast-1-4-5-8_vs_7         |   |
| 16 | ecoli-0-1_vs_2-3-5        | 39 | glass0                 | 63 | segment0                       | 87 | yeast-1_vs_7               |   |
| 17 | ecoli-0-1_vs_5            | 40 | glass1                 | 64 | shuttle-2_vs_5                 | 88 | yeast-2_vs_4               |   |
| 18 | ecoli-0-2-3-4_vs_5        | 41 | glass2                 | 65 | shuttle-6_vs_2-3               | 89 | yeast-2_vs_8               |   |
| 19 | ecoli-0-2-6-7_vs_3-5      | 42 | glass4                 | 66 | shuttle-c0-vs-c4               | 90 | yeast1                     |   |
| 20 | ecoli-0-3-4-6_vs_5        | 43 | glass5                 | 67 | shuttle-c2-vs-c4               | 91 | yeast3                     |   |
| 21 | ecoli-0-3-4-7_vs_5-6      | 44 | glass6                 | 68 | vehicle0                       | 92 | yeast4                     |   |
| 22 | ecoli-0-3-4_vs_5          | 45 | habermanImb            | 69 | vehicle1                       | 93 | yeast5                     |   |
| 23 | ecoli-0-4-6_vs_5          | 46 | iris0                  | 70 | vehicle2                       | 94 | yeast6                     |   |
| 24 | ecoli-0-6-7_vs_3-5        | 47 | kr-vs-k-one_vs_fifteen | 71 | vehicle3                       | 95 | zoo-3                      |   |


## One-Class Classifiers: 

* Average Absolute Deviation Linear Method for Deviation-based Outlier Detection
* Fully connected AutoEncoder
* Average Feature Bagging
* Average KNN
* Bagging-Random Miner [3]
* Clustering-Based Local Outlier Factor
* Copula-Based Outlier Detection
* Connectivity-Based Outlier Factor
* Elliptic Envelope
* Gaussian Mixture Model
* Histogram-based Outlier Detection
* Interquartile Range Linear Method for Deviation-based Outlier Detection
* Isolation Forest
* Largest KNN
* Lightweight On-line Detector of Anomalies
* Identifying Density-Based Local Outliers
* Locally Selective Combination in Parallel Outlier Ensembles
* Maximum Bagging
* Minimum Covariance Determinant
* Median KNN
* Multiple-Objective Generative Adversarial Active Learning
* One-Class K-means with Randomly-Projected Features Algorithm [4]
* One-Class Support Vector Machines
* Principal Component Analysis
* Sub-Space Outlier Detection
* Single-Objective Generative Adversarial Active Learning
* Variational Autoencoder
* Variance Linear Method for Deviation-Based Outlier Detection
* Extreme Boosting Based Outlier Detection

## Results

### AUC Results

![alt text](https://github.com/ML-Group-Col/One-Class-models/blob/main/Analysis/auc_ns.png)

### CD Nemenyi Test Diagram

![alt text](https://github.com/ML-Group-Col/One-Class-models/blob/main/Analysis/cd_ns.png)

### 2D Plot Comparing Ranks

![alt text](https://github.com/ML-Group-Col/One-Class-models/blob/main/Analysis/noscaler_rank.png)

## Libraries

- Sci-kit - Models [5]
- PyOD - Models [6]
- Pandas - DataFrame 
- Matplotlib - Plots

## Authors

Daniela Gomez Cravioto - (https://github.com/danisha20)
Ramon Díaz Ramos - https://github.com/ramon_diaz)
Michael Zenkl - (https://github.com/ToxicFyre)

## References

[1] J. Alcalá-Fdez, A. Fernandez, J. Luengo, J. Derrac, S. García, L. Sánchez, F. Herrera. KEEL Data-Mining Software Tool: Data Set Repository, Integration of Algorithms and Experimental Analysis Framework. Journal of Multiple-Valued Logic and Soft Computing 17:2-3 (2011) 255-287.

[2] DemSar, Janez. 2006. “Statistical Comparisons of Classifiers over Multiple Data Sets.” Journal of Machine Learning Research 7: 1–30

[3] Camiña, J.B., Medina-Pérez, M.A., Monroy, R. et al. Bagging-RandomMiner: a one-class classifier for file access-based masquerade detection. Machine Vision and Applications 30, 959–974 (2019). https://doi.org/10.1007/s00138-018-0957-4

[4] M. E. Villa-Pérez and L. A. Trejo, "m-OCKRA: An Efficient One-Class Classifier for PersonalRisk Detection, Based on Weighted Selection of Attributes", IEEE Access vol. 8, pp. 41749-41763, Feb. 2020.

[5] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

[6] Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
