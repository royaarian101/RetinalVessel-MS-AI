This is a code for vessel segmentation:

The folder named all  contains the SLO images. 
#### how to run : (run the following codes in order)
 1. preprocessing stage: run vessel_preprocessing.ipynb,
   matching the SLO images with a reference image.
   this return vessel_preprocessed file that contain the matched SLO images

2. Run vessel_automorph.ipynb on colab ,  
  zip vessel_preprocessed file as an input file of images.
  Return resize and resize_binary file containing gray and binary vessel segments.

3- Run vessel_segmentation.ipynb , analyzing vessel object candidates by using region growing algorithm to solve dis-connectivity in blood vessels and then removing object vessels with the area lower than 50.
  this returns final_vessel_result file containing the final binary vessel tree in the SLO images 
   
