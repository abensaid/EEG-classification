

function [ClassificationAccuracy, PredictedClass, PredictedClassNumber] = ...
        Knn_Classifier_ConventionalGeneral(EEGData, Knn, NData, FeatureExtFlag)

    if FeatureExtFlag == 0
        NClasses = 4;

        groups = zeros(NData * NClasses, 1);
        groups(1:NClasses:end) = 0;
        groups(2:NClasses:end) = 1;
        groups(3:NClasses:end) = 2;
        groups(4:NClasses:end) = 3;

        groups=groups';

        cp_nb = classperf(groups);
        classes = Knn.predict(EEGData);
        ClassIdx = mode(classes);

        if isnan(ClassIdx)
            ClassificationAccuracy = 0;
            PredictedClass = 'None';
            PredictedClassNumber = 4;
        else
            PredictedClassNumber = ClassIdx;

            TestIdx = zeros(NData*NClasses, 1);
            TestIdx(ClassIdx+1:NClasses:end) = 1;
            TestIdx = logical(TestIdx);

            classperf(cp_nb, classes, TestIdx);
            ClassificationAccuracy = cp_nb.CorrectRate*100;

            if PredictedClassNumber == 0
                PredictedClass = 'Left';
            elseif PredictedClassNumber == 1
                PredictedClass = 'Right';
            elseif PredictedClassNumber == 2
                PredictedClass = 'Up';
            elseif PredictedClassNumber == 3
                PredictedClass = 'Down';
            else
                PredictedClass = 'None';
            end
        end
        

    else
        NClasses = 4;
        data = zeros(14, 18);
        EEGData = EEGData';
        
        % Feature Selection
        % ===================

        index = 1;

        for i=1:14

            a = EEGData(i, :);
            mav=sum(abs(a))/NData;
            wavelen=0;
            for k=2:NData
                 wavelen=wavelen+sum(a(k)-a(k-1));
            end
            var=(sum(a.^2))/(NData-1);
            logdet=exp((sum(log10(abs(a))))/NData);
            ar_coeffs=aryule(a,9);
            data(index,:)=[max(a) min(a) mean(a) std(a) mav wavelen var logdet ar_coeffs];
            index=index+1;

        end

        groups = zeros(14 * NClasses, 1);
        groups(1:NClasses:end) = 0;
        groups(2:NClasses:end) = 1;
        groups(3:NClasses:end) = 2;
        groups(4:NClasses:end) = 3;

        groups=groups';

        cp_Knn = classperf(groups);
        classes = Knn.predict(data);
        ClassIdx = mode(classes);

        if isnan(ClassIdx)
            ClassificationAccuracy = 0;
            PredictedClass = 'None';
            PredictedClassNumber = 4;
        else
            PredictedClassNumber = ClassIdx;

            TestIdx = zeros(14*NClasses, 1);
            TestIdx(ClassIdx+1:NClasses:end) = 1;
            TestIdx = logical(TestIdx);

            classperf(cp_Knn, classes, TestIdx);
            ClassificationAccuracy = cp_Knn.CorrectRate*100;

            if PredictedClassNumber == 0
                PredictedClass = 'Left';
            elseif PredictedClassNumber == 1
                PredictedClass = 'Right';
            elseif PredictedClassNumber == 2
                PredictedClass = 'Up';
            elseif PredictedClassNumber == 3
                PredictedClass = 'Down';
            else
                PredictedClass = 'None';
            end
        end


        
    end
    
       
 end