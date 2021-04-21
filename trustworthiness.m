function [trustworthiness,xObs,xRand] = trustworthiness(labels, labelValues, metric, iters, plotFlag)
    %{
    Durán, C., Ciucci, S., Palladini, A. et al. Nonlinear machine learning pattern 
    recognition and bacteria-metabolite multilayer network analysis of perturbed 
    gastric microbiome. Nat Commun 12, 1926 (2021). 
    https://doi.org/10.1038/s41467-021-22135-x

    This program will compute a trustworthiness pvalue given a metric,
    the labels of a dataset and their respective values (also known as scores).
 
    The metrics available are:
    Mann-Witheny Pvalue ('MWpval')
    Permutation test by mean ('permmean')
    Permutation test by median ('permmedian')
    Area under the ROC-Curve ('AUC')
    Area under the Precision-Recall curve ('AUPR')
    Matthews Correlation Coefficient ('MCC')
    F-score ('Fscore')
    Accuracy ('accuracy')
    Pearson correlation('pearscorr')
    Spearman correlation('spearcorr')


    Inputs

    -labels: integer numeric vector with the sample label information.
    example: [1 1 1 1 2 2 2 1 2 2 2 1 2 2 1]

    -labelValues: floating numeric vector with sample scores. The ordering
    should correspond with the labels variable.
    example: [0.12 0.42 0.26 0.64 0.75 0.83 0.23 0.75 0.74 0.45 0.44 0.87 0.54 0.60 0.09]

    -metric: string referring to the metric for which to compute the
    trustworthiness.
    example: 'AUC'

    Optional inputs:

    -iters: Numeric value for iterations to create the null model.
    Default: 1000

    -plotFlag: binary value for the generation of the distribution plot after 
    trustworthiness calculation
    Default: true

    Outputs:

    -trustworthiness: trustworthiness pvalue calculated from the null model
    taking as reference the observable input value xObs.

    -xObs: observable computed value from the respective metric
    example: 0.62 (calculated as the AUC of the example labels and labelValues)

    -xRand: null model with all the metric performances from the resampling 
    approach. Its length is the same as the iters variable value.
    %}
    
    %initializing variables
    narginchk(3,5)
    if nargin < 4 || isempty(iters), iters = 1000; end
    if nargin < 5, plotFlag = true; end
    
    %converting labels to numeric 1 and 2 if labels variable was not
    %provided like it for biclass metrics (MWpval, AUC, AUPR, Fscore, MCC). 
    %Accuracy is here restricted for 2 binary problems as well. 
    uLabels = unique(labels);
    if length(uLabels) == 2
        if ~(uLabels(1) == 1 && uLabels(2) == 2)
            newLabels = ones(length(labels),1);
            newLabels(labels == uLabels(2)) = 2;
            labels = newLabels;
        end
    end
    
    %obtaining observable variable
    switch metric
        case 'MWpval'
            xObs = ranksum(labelValues(labels==1), labelValues(labels==2));
            if isnan(xObs)
                xObs = 1;
            end
        case 'permmean'
            xObs = permutation_test(labelValues(labels==1), labelValues(labels==2), 'mean', 'both', 1000, 1);
        case 'permmedian'
            xObs = permutation_test(labelValues(labels==1), labelValues(labels==2), 'median', 'both', 1000, 1);
        case 'AUC'
            xObs = compute_AUC(labelValues(labels==1), labelValues(labels==2));
        case 'AUPR'
            xObs = max(compute_AUPR(labelValues, double(labels==1)), compute_AUPR(labelValues, double(labels==2)));
        case 'MCC'
            labels_ = num2cell(double(labels==1));
            labels_ = cellfun(@num2str,labels_,'uni',0);
            xObs = computeFromCM(labels_, labelValues, labels_{1},metric);
        case 'pearscorr'
            if isrow(labelValues); labelValues = labelValues'; end
            if isrow(labels); labels = labels'; end
            xObs = corr(labelValues, labels, 'type', 'Pearson');
        case 'spearcorr'
            if isrow(labelValues); labelValues = labelValues'; end
            if isrow(labels); labels = labels'; end
            xObs = corr(labelValues, labels, 'type', 'Spearman');
        case 'Fscore'
            labels_ = num2cell(double(labels==1));
            labels_ = cellfun(@num2str,labels_,'uni',0);
            xObs = computeFromCM(labels_, labelValues, labels_{1},metric);
        case 'accuracy'
            labels_ = num2cell(double(labels==1));
            labels_ = cellfun(@num2str,labels_,'uni',0);
            xObs = computeFromCM(labels_, labelValues, labels_{1},metric);
        otherwise
            error('Possible metrics: ''MWpval'', ''permmean'', ''permmedian'', ''MCC'', ''AUC'', ''AUPR'',''accuracy'' ,''Fscore'', ''pearscorr'', ''spearcorr''.');
    end
    
    %creating the null distribution in x_rand variable
    rstr = RandStream('mt19937ar','Seed',1);
    xRand = zeros(iters,1);
    for i = 1:iters
        values_perm = labelValues(randperm(rstr,length(labelValues)));
        xRand(i) = compute_metric(labels, values_perm, metric);
    end

    %computing pvalue trustworthiness
    if any(strcmp(metric, {'MWpval', 'permmean', 'permmedian'}))
        trustworthiness = (sum(xRand <= xObs) + 1) / (iters + 1);
    elseif any(strcmp(metric, {'AUC', 'AUPR', 'MCC', 'pearscorr', 'spearcorr', 'Fscore','accuracy'}))
        trustworthiness = (sum(xRand >= xObs) + 1) / (iters + 1);
    end
    
    %plotting if flag true
    if plotFlag
        plotDistribution(trustworthiness,xRand,xObs,metric);
    end
end

function result = compute_metric(label, labelValues, metric)

    if strcmp(metric, 'MWpval')
        result = ranksum(labelValues(label==1), labelValues(label==2));
        if isnan(result)
            result = 1;
        end
    elseif strcmp(metric, 'permmean')
        result = permutation_test(labelValues(label==1), labelValues(label==2), 'mean', 'both', 1000, 1);
    elseif strcmp(metric, 'permmedian')
        result = permutation_test(labelValues(label==1), labelValues(label==2), 'median', 'both', 1000, 1);
    elseif strcmp(metric, 'AUC')
        result = compute_AUC(labelValues(label==1), labelValues(label==2));
    elseif strcmp(metric, 'AUPR')
        result = max(compute_AUPR(labelValues, double(label==1)), compute_AUPR(labelValues, double(label==2)));
    elseif strcmp(metric, 'MCC')
        labels_ = num2cell(double(label==1));
        labels_ = cellfun(@num2str,labels_,'uni',0);
        result = computeFromCM(labels_, labelValues, labels_{1},metric);
    elseif strcmp(metric, 'Fscore')
        labels_ = num2cell(double(label==1));
        labels_ = cellfun(@num2str,labels_,'uni',0);
        result = computeFromCM(labels_, labelValues, labels_{1},metric);
    elseif strcmp(metric, 'accuracy')
        labels_ = num2cell(double(label==1));
        labels_ = cellfun(@num2str,labels_,'uni',0);
        result = computeFromCM(labels_, labelValues, labels_{1},metric);
    elseif strcmp(metric, 'pearscorr')
        if isrow(labelValues); labelValues = labelValues'; end
        if isrow(label); label = label'; end
        result = corr(labelValues, label, 'type', 'Pearson');
    elseif strcmp(metric, 'spearcorr')
        if isrow(labelValues); labelValues = labelValues'; end
        if isrow(label); label = label'; end
        result = corr(labelValues, label, 'type', 'Spearman');
    else
        error('Possible metrics: ''ranksum'', ''permmean'', ''permmedian'', ''AUC'', ''AUPR'' , ''MCC'', ''pearscorr'', ''spearcorr''.');
    end
end

function AUC = compute_AUC(m1, m0)
    if isrow(m1); m1 = m1'; end
    if isrow(m0); m0 = m0'; end

    scores = [m1; m0];
    labels = [ones(size(m1)); zeros(size(m0))];
    [~,~,~,AUC] = perfcurve(labels, scores, 1);
    if AUC < 0.5
        AUC = 1 - AUC;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function AUPR = compute_AUPR(scores, labels)
    %%% INPUT %%%
    % scores - numerical scores for the samples
    % labels - binary labels indicating the positive and negative samples

    %%% OUTPUT %%%
    % aupr - area under precision-recall

    validateattributes(scores, {'numeric'}, {'vector','finite'})
    n = length(scores);
    validateattributes(labels, {'numeric'}, {'vector','binary','numel',n})
    if isrow(scores); scores = scores'; end
    if isrow(labels); labels = labels'; end

    [scores,idx] = sort(-scores, 'ascend');
    labels = labels(idx);
    [~,ut,~] = unique(scores);
    ut = [ut(2:end)-1; n];
    tp = full(cumsum(labels));
    recall = tp ./ sum(labels);
    precision = tp ./ (1:n)';
    recall = recall(ut);
    precision = precision(ut);
    if all(recall==1)
        AUPR = precision(1);
    else
        AUPR = trapz(recall,precision) / (1-recall(1));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function reslt = computeFromCM(labels, scores, positiveClass, metric)
    if isrow(scores); scores = scores'; end
    if isrow(labels); labels = labels'; end

    % total amount of positive labels
    totalPositive = sum(strcmp(labels, positiveClass));
    % total amount of negative labels
    totalNegative = sum(~strcmp(labels, positiveClass));

    % identifying the negative class
    negativeClass = unique(labels(~strcmp(labels, positiveClass)));

    % sort the scores and obtained the sorted indices
    [~, idxs] = sort(scores);

    % sort the original labels according to the sorted scores
    trueLabels = labels(idxs);

    for in=1:2
        % since the position of the clusters is unknown
        % we take as comparison a perfect segregation in both ways
        % positive cluster in the left side, and negative cluster in the right side,
        % and viseversa
        switch in
            case 1
                predictedLabels = [repmat({positiveClass},totalPositive,1);repmat(negativeClass,totalNegative,1)];
            case 2
                predictedLabels = [repmat(negativeClass,totalNegative,1);repmat({positiveClass},totalPositive,1)];
        end

        % clasifiers
        TP = sum(ismember(predictedLabels, positiveClass) & ismember(trueLabels, positiveClass));
        TN = sum(ismember(predictedLabels, negativeClass) & ismember(trueLabels, negativeClass));
        FP = sum(ismember(predictedLabels, positiveClass) & ismember(trueLabels, negativeClass));
        FN = sum(ismember(predictedLabels, negativeClass) & ismember(trueLabels, positiveClass));

        switch metric
            case 'MCC'
                if ((TP == 0 && FP == 0) || (TN == 0 && FN == 0))
                    val(in) = 0;
                else
                    % compute the Matthews Correlation Coefficient (MCC)
                    val(in) = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
                end
            case 'Fscore'
                prec = TP/(TP+FP); %PPV
                if isnan(prec), prec = 0; end
                sens = TP/(TP+FN); %Recall
                if isnan(sens), sens = 0; end
                
                if prec == 0 && sens == 0
                    val(in) = 0;
                else
                    val(in) = (2*prec*sens)/(prec+sens); 
                end
            case 'accuracy'
                val = (TP + TN) / (TP + TN + FP + FN);
        end
    end
    % select the best metric side
    reslt = max([val(:)]);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p = permutation_test(x1, x2, func_name, tail_type, iters, rstr_flag)

    %%% INPUT %%%
    % x1, x2 - numerical vectors
    % func_name - name of the function to evaluate the statistic for each vector (examples: 'mean', 'median')
    % tail_type - choose between:
    %             'right' -> test for statistic in x1 greater than in x2
    %             'left'  -> test for statistic in x1 lower than in x2
    %             'both'  -> test for statistic in x1 greater or lower than in x2
    % iters - number of iterations (if not given or empty, default = 1000)
    % rstr_flag - 1 or 0 to indicate if a random stream generator should be used or not
    %             for reproducibility of the results (if not given or empty, default = 0)

    %%% OUTPUT %%%
    % p - empirical p-value obtained using the formula discussed in:
    %     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC379178/

    % check input
    narginchk(3,6)
    validateattributes(x1, {'numeric'}, {'vector','finite'});
    validateattributes(x2, {'numeric'}, {'vector','finite'});
    validateattributes(func_name, {'char'}, {});
    validateattributes(tail_type, {'char'}, {});
    if ~exist('tail_type', 'var') || isempty(tail_type)
        tail_type = 'both';
    elseif ~any(strcmp(tail_type,{'right','left','both'}))
        error('Invalid value for ''tail_type''.')
    end
    func = str2func(func_name);
    if ~exist('iters', 'var') || isempty(iters)
        iters = 1000;
    else
        validateattributes(iters, {'numeric'}, {'scalar','integer','positive'});
    end
    if ~exist('rstr_flag', 'var') || isempty(rstr_flag)
        rstr_flag = 0;
    else
        validateattributes(rstr_flag, {'numeric'}, {'scalar','binary'});
        if rstr_flag
            rstr = RandStream('mt19937ar','Seed',1);
        end
    end
    if isrow(x1)
        x1 = x1';
    end
    if isrow(x2)
        x2 = x2';
    end

    % observed difference
    diff_obs = func(x1) - func(x2);

    % random distribution of differences
    x = [x1; x2];
    n = length(x);
    n1 = length(x1);
    diff_rand = zeros(1,iters);
    for i = 1:iters
        if rstr_flag
            rp = randperm(rstr, n);
        else
            rp = randperm(n);
        end
        xr1 = x(rp(1:n1));
        xr2 = x(rp(n1+1:end));
        diff_rand(i) = func(xr1) - func(xr2);
    end

    % p-value
    if strcmp(tail_type, 'right')
        p = (sum(diff_rand >= diff_obs) + 1) / (iters + 1);
    elseif strcmp(tail_type, 'left')
        p = (sum(diff_rand <= diff_obs) + 1) / (iters + 1);
    elseif strcmp(tail_type, 'both')
        p = (sum(abs(diff_rand) >= abs(diff_obs)) + 1) / (iters + 1);
    end
end

function plotDistribution(trustworthiness,nullModel,xObs,metric)
    %bandWidth is use to smoothen the distribution curve
    bandWidth = 0.05;
    
    %q is used to mark the 0.05 (upper or lower, depending on the metric)
    %tail in the distribution
    if any(strcmp(metric, {'MWpval', 'permmean', 'permmedian'}))
        q = quantile(nullModel,0.05);
    else    
        q = quantile(nullModel,0.95); 
    end
    figure;
    suptitle('Null Model Distribution')
    [f,xi] = ksdensity(nullModel,'Bandwidth',bandWidth);
    plot(xi,f,'LineWidth',2);
    ylabel('Density')
    xlabel([metric ' values'])
    %xlim([0,1])
    title(['Trustworthiness P-value: ' num2str(trustworthiness)])
    hold on
    plot([xObs,xObs],[0,0],'Marker','*','MarkerSize',10);
    plot([q,q],[0,max(f)]);
    hold off
end
