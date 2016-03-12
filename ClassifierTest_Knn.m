
% ****************************************** %
% ***** Training The Classifier ************ %
% ****************************************** %
clear
clc


% Parameters
% ================
FeatureExtFlag = 0;         % 1 to Classify with Feature extraction.

if FeatureExtFlag == 1
    NData = 128 * 10;       % 128 represents 1 seconds
else
    NData = 128 * 40;
end

NTestData = 128*10;
NClasses = 4;
TotalNData = 6400;
% ================================================================================

KnnClassifierTraining(FeatureExtFlag, NData);              % Training

load('Knn');
Knn.NumNeighbors = 13;

FileName = sprintf('./MouthData/Up.txt');
FID = fopen(FileName, 'r');
Data = fscanf(FID, '%f');
fclose(FID);

DataTest = reshape(Data, length(Data)/TotalNData, TotalNData)';
DataTest = DataTest(NData+1 : NData + NTestData, :);

[ClassificationAccuracy, PredictedClass, PredictedClassNumber] = ...
        Knn_Classifier_ConventionalGeneral(DataTest, Knn, NTestData, FeatureExtFlag)









