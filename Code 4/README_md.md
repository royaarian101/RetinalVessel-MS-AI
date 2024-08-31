this code compute feature iomportance for classifying MS and HC subjects.

features_SLO is a dictionary in which its key defines the number of subjects and its value is an array representing clinical features for IR-SLO images for the corresponding subject. 
label_SLO is a dictionary in which its keys defines the nember of subjects and its value represents the corresponding label for each subject(0 for HC and 1 for MS subjectr). 

SHAP_feature_importance_IR_SLO_images.ipynb:
First, the best features were selected by using a statistical method named mann-whithney, then the importance of the selected features was computed using SHAP feature importance approach. To select what classifier should be used for measuring FI, 3 conventional ML classifiers , SVM, RF and XGBoost ML models, were applied on the selected features. Then, the SHAP method was applied on the training, validation , internal and external test sets, and the FI score for the selected features was analyzed. 

FI_robustness_assessment.ipynb:
To assesee model robustness, again FI for the clinical features was computed without using Mann-whithney approach. SHAP FI method was applied for all the 4 data sets, training, validation, internal test and external test, and then  FI scores for the clinical features were analyzed.


