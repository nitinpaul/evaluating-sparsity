# Coarse-to-Fine: Learning Compact Discriminative Representation for Single-Stage Image Retrieval
*Authors: Yunquan Zhu, Xinkai Gao, Bo Ke, Ruizhi Qiao, Xing Sun*

* **Summary:** The paper introduces a novel coarse-to-fine framework for learning compact and discriminative representations for single-stage image retrieval. This framework consists of two stages: a coarse stage that generates an initial low-dimensional representation of the image, and a fine stage that refines this representation using more local and discriminative features. Additionally, the paper proposes a novel loss function called MadaCos, which dynamically adjusts its scale and margin to improve the discriminativeness of the learned representations. Finally, the authors propose a hard negative sampling strategy that selects prominent local descriptors to focus on during training.
* **Implementation details:** ResNet50 and ResNet101 are mainly used for experiments. Models in this paper are initialized from Imagenet pre-trained weights, and are trained on the Google landmarks dataset V2 (GLDv2-clean).
* **Evaluation metrics:** The authors evaluate their method on several single-stage image retrieval benchmarks, including Recall@K, mAP, and NMI.
* **Datasets:** The evaluations are performed on Revisited Oxford and Revisited Paris datasets
* **Performance:** On the Hard variant of the two datasets, the authors demonstrate a mAP of 64.84% and 81.68% for Revisited Oxford and Revisited Paris respectively.
