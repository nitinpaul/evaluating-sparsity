# Global Features are All You Need for Image Retrieval and Reranking
*Authors: Shihao Shao, Kaifeng Chen, Arjun Karpur, Qinghua Cui, André Araujo, Bingyi Cao*

* **Summary:** This paper proposes SuperGlobal, a system that uses only global features for both retrieval and reranking stages, which is computationally more efficient than traditional methods that use local features in the reranking stage.
* **Implementation details:** The proposed methods can be applied to any model architecture, and in the paper, the authors apply their methods to CVNet \cite{lee2022correlation}. For the evaluation, I will apply the proposed method to the ResNet50 and ResNet101 models for consistency.
* **Evaluation metrics:** The paper primarily uses mAP for the evaluation
* **Datasets:** The evaluations are performed on Revisited Oxford and Revisited Paris datasets
* **Performance:** On the Hard variant of the two datasets, the authors demonstrate a mAP of 72.1% and 83.5% for Revisited Oxford and Revisited Paris respectively.
