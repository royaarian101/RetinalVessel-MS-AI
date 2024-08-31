# RetinalVessel-MS-AI
**Interpretable AI for Analyzing Retinal Vessel Morphology in Multiple Sclerosis Using IR-SLO Images**

Authors: Asieh Soltanipour, Roya Arian, Ali Aghababaei, Raheleh Kafieh

This project explores the identification of multiple sclerosis (MS)-specific features within infrared reflectance scanning laser ophthalmoscopy (IR-SLO) images. The work involves segmenting IR-SLO images into distinct anatomical structures, extracting clinically relevant features from these structures, and assessing their significance in differentiating between MS patients and healthy controls (HC).

For this project, we leveraged the code provided in the publication by Zhou et al., titled **"AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline"**


**Folder Structure**

**code_1: Optic Disc and Cup Segmentation**

This folder contains Python scripts for segmenting the optic disc and cup from IR-SLO images. The deep learning algorithm, originally developed for anatomical segmentation in color fundus images, has been adapted and optimized for IR-SLO images. Pre-processing and post-processing techniques are employed to enhance the accuracy of the segmentation.

**code_2: Vessel Segmentation**

This folder includes Python scripts for segmenting blood vessels in IR-SLO images. The deep learning algorithm has been customized with targeted pre- and post-processing stages to improve segmentation performance on IR-SLO images, similar to the approach used for optic disc and cup segmentation.

**code_3: Clinical Feature Extraction**

This folder contains Python scripts for extracting clinical features from the segmented anatomical structures obtained from code_1 and code_2. Extracted features include measurements related to the optic disc, optic cup, and blood vessels, which are crucial for subsequent analysis.

**code_4: Feature Importance Measurement**

This folder focuses on evaluating the importance of each clinical feature for differentiating between MS and healthy controls (HC). The most discriminative features are first identified using the Mann-Whitney U test, a statistical method for detecting significant differences between two groups. The SHAP (SHapley Additive exPlanations) method is then applied to assess the importance of these features.




**Workflow**

Segmentation:

Run the scripts in code_1 to segment the optic disc and cup from IR-SLO images.
Run the scripts in code_2 to segment blood vessels from the same images.


Feature Extraction:

Use the scripts in code_3 to extract clinical features from the segmented anatomical structures.
Feature Selection and Importance Measurement:

Execute the scripts in code_4 to perform the Mann-Whitney U test for feature selection.
Apply the SHAP method to the selected features to determine their importance in distinguishing between MS and HC.



**Requirements**

Python 3.x
Deep learning framework (TensorFlow)
Libraries: NumPy, OpenCV, scikit-learn, SHAP, and any other dependencies specified in the individual code files.


**Usage**

Ensure all dependencies are installed.
Place the input IR-SLO images in the designated directory.
Follow the workflow steps to segment the images, extract features, and evaluate their importance.


**Conclusion**

This project provides a comprehensive pipeline for segmenting IR-SLO images, extracting clinically relevant features, and evaluating their significance in the classification of MS. By combining deep learning-based segmentation with statistical feature selection methods, the pipeline enables robust analysis and the potential identification of MS-specific features within IR-SLO images.

For any questions or further assistance, please contact "asieh.soltanipour1365@gmail.com" or "royaarian101@gmail.com".



**Please ensure to include the following citations when utilizing any part of the code:**

[1] Soltanipour A, Arian R, Aghababaei A, Kafieh R, Ashtari F. Analyzing morphological alternations of vessels in multiple Sclerosis using SLO images of the eyes [Internet]. bioRxiv. 2023. Available from: http://dx.doi.org/10.1101/2023.12.12.23299846

[2] Arian, R., Aghababaei, A., Soltanipour, A., Khodabandeh, Z., Rakhshani, S., Iyer, S. B., Ashtari, F., Rabbani, H., & Kafieh, R. (2024). SLO-net: Enhancing multiple sclerosis diagnosis beyond optical coherence tomography using infrared reflectance scanning laser ophthalmoscopy images. Translational Vision Science & Technology, 13(7), 13. https://doi.org/10.1167/tvst.13.7.13

[3] Aghababaei A, Arian R, Soltanipour A, Ashtari F, Rabbani H, Kafieh R. Discrimination of Multiple Sclerosis using Scanning Laser Ophthalmoscopy Images with Autoencoder-Based Feature Extraction. Multiple Sclerosis and Related Disorders. 2024 Aug 1;88:105743–3.

[4] Zhou Y, Wagner SK, Chia MA, Zhao A, Woodward-Court P, Xu M, et al. AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline. Translational Vision Science & Technology. 2022 Jul 8;11(7):12. DOI: 10.1167/tvst.11.7.12.
