# Learning Spatial-context-aware Global Visual Feature Representation for Instance Image Retrieval
*Authors: Zhongyan Zhang, Lei Wang, Luping Zhou, Piotr Koniusz*

* **Summary:** The paper proposes a novel method for learning spatial-context-aware global visual features for instance image retrieval. This method uses a CNN backbone to learn visual features and a spatial context branch to learn spatial context information. The spatial context branch includes two modules: online token learning and distance encoding. These modules are used to encode what kind of surrounding local descriptors are present and their spatial distribution. The visual and spatial context information are then fused together to form a final, global feature representation.
* **Implementation details:** ResNet50 and ResNet101 are used as the CNN backbone for the models and are trained on the Google landmarks dataset V2 (GLDv2-clean)
* **Evaluation metrics:** The paper uses retrieval accuracy (Recall@K) as the primary evaluation metric. Additionally, it reports mean Average Precision (mAP) and normalized Discounted Cumulative Gain (nDCG) for a more comprehensive evaluation
* **Datasets:** The evaluations are performed on Revisited Oxford and Revisited Paris datasets
* **Performance:** The proposed method achieves significant improvements in retrieval performance compared to other methods on both the ROxford and RParis datasets. For example, on the ROxford dataset, the proposed method achieves a Recall@1 of 91.3%, compared to 83.8% for the best baseline method.
