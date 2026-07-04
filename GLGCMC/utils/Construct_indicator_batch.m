function Construct_indicator_batch(Data_list, dataRoot, ratioList, foldNum, ensureOneView)
% Construct incomplete-view indicator matrices for multiple datasets.
%
% Output:
%   <this file folder>/ration/<dataset>/<dataset>ratio10.mat
%   <this file folder>/ration/<dataset>/<dataset>ratio20.mat
%   ...
%
% Each saved MAT file contains:
%   folds{1:foldNum}: N x V indicator matrices
%       1 means the sample-view pair exists
%       0 means the sample-view pair is missing
%   N, V, ratio, viewDims, viewSizes, orientation and missing-rate records

clc;

if nargin < 1 || isempty(Data_list)
    Data_list = {
        'MSRCv1.mat'
        'Mfeat.mat'
        'CCV.mat'
        'Hdigit.mat'
        'MNIST-4.mat'
        'ORL.mat'
%         'BDG2.mat'
    };
end

scriptDir = fileparts(mfilename('fullpath'));

if nargin < 2 || isempty(dataRoot)
    dataRoot = fullfile(scriptDir, '..', 'data');
end

if nargin < 3 || isempty(ratioList)
    ratioList = 0.1:0.1:0.9;
end

if nargin < 4 || isempty(foldNum)
    foldNum = 10;
end

if nargin < 5 || isempty(ensureOneView)
    % Keep the same behavior as BuildIndicator_my by default.
    % Set true if every sample must keep at least one observed view.
    ensureOneView = false;
end

resultRoot = fullfile(scriptDir, 'ration');
if ~exist(resultRoot, 'dir')
    mkdir(resultRoot);
end

fprintf('Indicator result root: %s\n', resultRoot);

for dataIdx = 1:numel(Data_list)
    Data_name = Data_list{dataIdx};
    dataFile = resolve_dataset_file(Data_name, dataRoot);
    [~, datasetName, ~] = fileparts(Data_name);

    fprintf('\n====================================================\n');
    fprintf('Dataset: %s\n', Data_name);
    fprintf('File   : %s\n', dataFile);
    fprintf('====================================================\n');

    rawData = load(dataFile);
    [X_raw, dataVarName] = read_multiview_data(rawData, Data_name);
    [gt, labelVarName] = read_label_data(rawData, Data_name);
    N = numel(gt);

    [X, viewDims, viewSizes, orientation] = normalize_view_orientation(X_raw, N, Data_name);
    V = numel(X);

    datasetDir = fullfile(resultRoot, datasetName);
    if ~exist(datasetDir, 'dir')
        mkdir(datasetDir);
    end

    fprintf('Views: %d, samples: %d\n', V, N);
    fprintf('View dimensions: ');
    fprintf('%d ', viewDims);
    fprintf('\n');

    for ratioIdx = 1:numel(ratioList)
        ratio = ratioList(ratioIdx);
        ratioPercent = round(ratio * 100);

        folds = cell(1, foldNum);
        actualMissingRateByView = zeros(foldNum, V);
        actualMissingRateTotal = zeros(foldNum, 1);

        for iter = 1:foldNum
            Indicator = build_indicator_matrix(V, N, ratio, ensureOneView);
            folds{iter} = Indicator;

            actualMissingRateByView(iter, :) = sum(Indicator == 0, 1) / N;
            actualMissingRateTotal(iter) = sum(Indicator(:) == 0) / numel(Indicator);
        end

        saveFile = fullfile(datasetDir, [datasetName, 'ratio', num2str(ratioPercent), '.mat']);
        save(saveFile, ...
            'folds', ...
            'ratio', ...
            'ratioPercent', ...
            'foldNum', ...
            'N', ...
            'V', ...
            'viewDims', ...
            'viewSizes', ...
            'orientation', ...
            'Data_name', ...
            'dataFile', ...
            'dataVarName', ...
            'labelVarName', ...
            'ensureOneView', ...
            'actualMissingRateByView', ...
            'actualMissingRateTotal');

        fprintf('Saved ratio %.1f -> %s\n', ratio, saveFile);
    end
end

fprintf('\nAll indicator files have been saved under:\n%s\n', resultRoot);

end

function dataFile = resolve_dataset_file(Data_name, dataRoot)
    if exist(Data_name, 'file')
        dataFile = Data_name;
        return;
    end

    dataFile = fullfile(dataRoot, Data_name);
    if exist(dataFile, 'file')
        return;
    end

    error('Dataset file not found: %s', dataFile);
end

function [X_raw, dataVarName] = read_multiview_data(rawData, Data_name)
    if isfield(rawData, 'X')
        X_raw = rawData.X;
        dataVarName = 'X';
    elseif isfield(rawData, 'data')
        X_raw = rawData.data;
        dataVarName = 'data';
    else
        error('Dataset %s does not contain X or data.', Data_name);
    end

    if ~iscell(X_raw)
        error('Dataset %s: X/data must be a cell array.', Data_name);
    end
end

function [gt, labelVarName] = read_label_data(rawData, Data_name)
    if isfield(rawData, 'Y')
        gt = rawData.Y;
        labelVarName = 'Y';
    elseif isfield(rawData, 'y')
        gt = rawData.y;
        labelVarName = 'y';
    elseif isfield(rawData, 'gt')
        gt = rawData.gt;
        labelVarName = 'gt';
    elseif isfield(rawData, 'truelabel')
        gt = rawData.truelabel;
        labelVarName = 'truelabel';
    elseif isfield(rawData, 'label')
        gt = rawData.label;
        labelVarName = 'label';
    else
        error('Dataset %s does not contain Y/y/gt/truelabel/label.', Data_name);
    end

    if iscell(gt)
        gt = gt{1};
    end
    gt = double(gt(:));
end

function [X, viewDims, viewSizes, orientation] = normalize_view_orientation(X_raw, N, Data_name)
    X_raw = X_raw(:)';
    V = numel(X_raw);
    X = cell(1, V);
    viewDims = zeros(1, V);
    viewSizes = zeros(V, 2);
    orientation = cell(1, V);

    for v = 1:V
        Xi = full(double(X_raw{v}));
        viewSizes(v, :) = size(Xi);

        if size(Xi, 2) == N
            X{v} = Xi;
            orientation{v} = 'd_by_n';
        elseif size(Xi, 1) == N
            X{v} = Xi';
            orientation{v} = 'n_by_d_to_d_by_n';
        else
            error('Dataset %s view %d size [%d, %d] does not match sample number N=%d.', ...
                Data_name, v, size(Xi, 1), size(Xi, 2), N);
        end

        viewDims(v) = size(X{v}, 1);
    end
end

function Indicator = build_indicator_matrix(V, N, ratio, ensureOneView)
    if ratio < 0 || ratio > 1
        error('Missing ratio must be in [0, 1].');
    end

    Indicator = ones(N, V);
    missingNum = round(ratio * N);

    for v = 1:V
        if missingNum > 0
            missingIdx = randperm(N, missingNum);
            Indicator(missingIdx, v) = 0;
        end
    end

    if ensureOneView
        emptySamples = find(sum(Indicator, 2) == 0);
        for i = 1:numel(emptySamples)
            Indicator(emptySamples(i), randi(V)) = 1;
        end
    end
end
