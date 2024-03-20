# DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features
*Authors: Min Yang, Dongliang He, Miao Fan, Baorong Shi, Xuetong Xue, Fu Li, Errui Ding, Jizhou Huang*

* **Summary:** This paper proposes a single-stage approach instead of the traditional two-stage approach (fetching candidate images using global features and ranking them using local features). The paper is used as a SOTA benchmark for many of the other works listed here
* **Implementation details:** ResNet50 and ResNet101 are used as the CNN backbone for the models and are trained on the Google landmarks dataset V2 (GLDv2-clean)
* **Evaluation metrics:** The paper primarily uses mAP for the evaluation
* **Datasets:** The evaluations are performed on Revisited Oxford and Revisited Paris datasets
* **Performance:** On the Hard variant of the two datasets, the authors demonstrate an mAP of 61.1% and 80.3% for Revisited Oxford and Revisited Paris respectively
