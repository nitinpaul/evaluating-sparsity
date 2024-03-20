# Revisiting Self-Similarity: Structural Embedding for Image Retrieval
*Authors: Seongwon Lee, Suhyeon Lee, Hongje Seong, Euntai Kim*

* **Summary:** This work proposes Structural Embedding Network (SENet), an architecture that exploits the latent geometric structures in images in order to improve performance during the global retrieval stage. SENet captures the internal structures from various images and compresses them into dense self-similarity descriptors. A global embedding is obtained by fusing these descriptors along with original image features. The resulting representation incorporates both geometric and visual cues of the image, and is robust to look-alike image pairs.
* **Implementation details:** ResNet50 and ResNet101 are used as the CNN backbone for the models and are trained on the Google landmarks dataset V2 (GLDv2-clean)
* **Evaluation metrics:** The paper primarily uses mAP for the evaluation
* **Datasets:** The evaluations are performed on Revisited Oxford and Revisited Paris datasets
* **Performance:** On the Hard variant of the two datasets, the authors demonstrate a mAP of 66% and 82.8% for Revisited Oxford and Revisited Paris respectively.
