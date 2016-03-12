

% This classifier considers feature selection on each channel. 
% ==============================================================
% In this file I take only 10 seconds from each channel and get the
% features in those 10 seconds. To change it, Change the NData paramter.

function KnnClassifierTraining(FeatureExtFlag, NData)

    % ****************************************** %
    % ***** Training The Classifier ************ %
    % ****************************************** %

    NClasses = 4;
    TotalNData = 6400;

%     FeatureExtFlag = 1;         % 1 to Train with Feature extraction.

    % Left
    % =======================   
    FileName = sprintf('./MouthData/Left.txt');
    FID = fopen(FileName, 'r');
    Data = fscanf(FID, '%f');
    fclose(FID);

    DataLeft = reshape(Data, length(Data)/TotalNData, TotalNData)';
    % =====================================================
    % Right
    % ======
    FileName = sprintf('./MouthData/Right.txt');
    FID = fopen(FileName, 'r');
    Data = fscanf(FID, '%f');
    fclose(FID);

    DataRight = reshape(Data, length(Data)/TotalNData, TotalNData)';
    % =====================================================
    % Up
    % ======
    FileName = sprintf('./MouthData/Up.txt');
    FID = fopen(FileName, 'r');
    Data = fscanf(FID, '%f');
    fclose(FID);

    DataUp = reshape(Data, length(Data)/TotalNData, TotalNData)';
    % =====================================================
    % Down
    % ======
    FileName = sprintf('./MouthData/Down.txt');
    FID = fopen(FileName, 'r');
    Data = fscanf(FID, '%f');
    fclose(FID);

    DataDown = reshape(Data, length(Data)/TotalNData, TotalNData)';
    % =====================================================


    if FeatureExtFlag == 0

        DataLeft = DataLeft(1:NData, :);
        DataRight = DataRight(1:NData, :);
        DataUp = DataUp(1:NData, :);
        DataDown = DataDown(1:NData, :);

        % interleaving the data for the classifier.
        data = zeros(NData * NClasses, 14);
        data(1:NClasses:end, :) = DataLeft;
        data(2:NClasses:end, :) = DataRight;
        data(3:NClasses:end, :) = DataUp;
        data(4:NClasses:end, :) = DataDown;

        groups = zeros(NData * NClasses, 1);
        groups(1:NClasses:end) = 0;
        groups(2:NClasses:end) = 1;
        groups(3:NClasses:end) = 2;
        groups(4:NClasses:end) = 3;

        groups=groups';

    else
        
        DataLeft = DataLeft(1:NData, :)';
        DataRight = DataRight(1:NData, :)';
        DataUp = DataUp(1:NData, :)';
        DataDown = DataDown(1:NData, :)';

        index = 1;
        groups = zeros(14 * NClasses, 1);
        data = zeros(14 * NClasses, 18);

        % Feature Extraction
        % ===================
        for i=1:14

            % Left
            % ------
            a = DataLeft(i, :);
            mav=sum(abs(a))/NData;
            wavelen=0;
            for k=2:NData
                 wavelen=wavelen+sum(a(k)-a(k-1));
            end
            var=(sum(a.^2))/(NData-1);
            logdet=exp((sum(log10(abs(a))))/NData);
            ar_coeffs=aryule(a,9);
            data(index,:)=[max(a) min(a) mean(a) std(a) mav wavelen var logdet ar_coeffs];
            groups(index) = 0;
            index=index+1;

            % Right
            % ------
            a = DataRight(i, :);
            mav=sum(abs(a))/NData;
            wavelen=0;
            for k=2:NData
                 wavelen=wavelen+sum(a(k)-a(k-1));
            end
            var=(sum(a.^2))/(NData-1);
            logdet=exp((sum(log10(abs(a))))/NData);
            ar_coeffs=aryule(a,9);
            data(index,:)=[max(a) min(a) mean(a) std(a) mav wavelen var logdet ar_coeffs];
            groups(index) = 1;
            index=index+1;

            % Up
            % ------
            a = DataUp(i, :);
            mav=sum(abs(a))/NData;
            wavelen=0;
            for k=2:NData
                 wavelen=wavelen+sum(a(k)-a(k-1));
            end
            var=(sum(a.^2))/(NData-1);
            logdet=exp((sum(log10(abs(a))))/NData);
            ar_coeffs=aryule(a,9);
            data(index,:)=[max(a) min(a) mean(a) std(a) mav wavelen var logdet ar_coeffs];
            groups(index) = 2;
            index=index+1;

            % Down
            % ------
            a = DataDown(i, :);
            mav=sum(abs(a))/NData;
            wavelen=0;
            for k=2:NData
                 wavelen=wavelen+sum(a(k)-a(k-1));
            end
            var=(sum(a.^2))/(NData-1);
            logdet=exp((sum(log10(abs(a))))/NData);
            ar_coeffs=aryule(a,9);
            data(index,:)=[max(a) min(a) mean(a) std(a) mav wavelen var logdet ar_coeffs];
            groups(index) = 3;
            index=index+1;

        end
    end

    % =====================================================

    [train, test] = crossvalind('holdOut',groups); 
    Knn = ClassificationKNN.fit(data, groups);

    save('Knn', 'Knn');
    % ****************************************** %






end

