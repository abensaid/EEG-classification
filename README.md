# EEG-classification
Knn General Classifier
-------------------------

This is used for both Knn with feature extraction from the time properties of each channel, and Knn 
without feature extraction (i.e. considering the 14 channels as the features).
EEG data (14 channels) are available in "MouthData" folder.

======================================================================================================

ClassifierTest_Knn.m
----------------------
This is the main file to run the code. 
	
	FeatureExtFlag 	        ==>  	1 for feature extraction classification and 0 for the other case.
	NTestData		==>	Number of Data taken to test the classifier. These Data is taken just after
					the NData taken for training.
	NData			==> 	Number of Data taken to train the classifier. 40 seconds in case of non-feature
					extraction, and 10 seconds in case of feature extraction.
	TotalNData		==>	Total Ndata captured for each movement.
	
======================================================================================================

KnnClassifierTraining.m
----------------------
File used for training the classifier according to the "FeatureExtFlag" value.

======================================================================================================

Knn_Classifier_ConventionalGeneral.m
--------------------------------------------

General Knn classifier that works for both feature and non-feature extraction methods.

======================================================================================================
