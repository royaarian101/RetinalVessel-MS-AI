It is a code for feature extraction from segmented disc, cup and blood vessels in IR-SLO images .
This code extracts clinical features for all existing SLO images of each subject.

HOW to RUN :
run feature_extraction.ipynb
This returns features_SLO.pkl and labels_SLO.pkl .
These two pickle files will be inputs for the next step.
 
features_SLO is a dictionary in which each key refers to one subject( MS or HC) and the value of each key is an array with size of ( number of images_per_subject x number of features(24 in this paper) ).
labels_SLO is a dictionary in which its key is the nember of each subject and its corresponding value defines the label of subject(0 for HC and 1 for MS).
