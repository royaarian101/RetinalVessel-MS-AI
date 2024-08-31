This is a code for OD  and CUP segmentation in the SLO images

#### how to run:

1. run OD_preprocessing_stage.ipynb, loading image and performing preprocessing steps .
 it returns  preprocessed_OD file containing preprocessed SLO images that was used as an  input to the LWnet in the auto-morph algorithm.  

2. run disc_candidates.ipynb on cpu in order to compute OD candidates
    -zip preprocessed_OD file 
    -run disc_candidates.ipynb
    -run the code and clone the auto-morph 
    -download disc_candidates_automorph and  place it in file  named M2_lwnet_disc_cup
    -change the name of images path in line 183 to preprocessed_OD 
    -saving the results of the second channel of decoder in LWnet as the OD candidate 
    -output file: OD_candidates

3. run OD_segmentation.ipynb : analyzing OD candidates and obtaining th right OD region. 
   -return 2 files named analyzed_OD_candidate and OD_boundary_segmentation that contain the OD candidates analyzed by region propsalgorithm and the OD boundary extracted by blob algorithm. 
   -return cup_pre_processed file including the widow surrounding OD in the SLO images

4.  run disc_candidates.ipynb on cpu in order to compute cup candidates
    -zip cup_pre_processed file 
    -run disc_candidates.ipynb
    -run the code and clone the auto-morph 
    -download disc_candidates_automorph and  place it in file  named M2_lwnet_disc_cup
    -change the name of images path in line 183 to cup_pre_processed
    -saving the results of the first channel of decoder in LWnet as the cup candidate(at line 123 of disc_candidates_automorph.py ) 
    -output file: cup_candidates, change the name of result file to cup_candidates,(at line 35 of disc_candidates_automorph.py)  

5- run cup_segmentation.ipynb , analyzing cup candidates and obtaining th right cup region. 
return cup_boundary file containing the SLO images with the boundary of cup  on them. 