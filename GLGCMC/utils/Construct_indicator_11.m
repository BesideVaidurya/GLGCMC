clear;clc;
addpath("datasets")
datasets='CCV';
load("CCV.mat")

% for iv = 1 : length(data)
%     X{iv} = data{iv};
% end
folds=cell(1,10);
% X{1} = sport01;
% X{2} = sport02;
for i = 1:1
    for ratio=0.1:0.1:0.9
%       for ratio=0.5
        if ~exist(['./Incomplete_index/', datasets])
            mkdir(['./Incomplete_index/', datasets]);
        end     
            for iter=1:10
            load(datasets); 
            X=X';
            Indicator = BuildIndicator_my(size(X, 2), size(X{1}, 1), ratio);
            folds{iter}=Indicator;
            end
            save(['./Incomplete_index/',datasets,'/',datasets, 'ratio', int2str(ratio*100),'.mat'], 'folds');
              
    end
end