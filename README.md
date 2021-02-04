# trustworthiness
MATLAB code for the resampling strategy to compute trustworthiness p-value

Trustworthiness exploits a resampling technique based on label-reshuffling to build a null model. The labels are reshuffled uniformly at random on the embedded points whose location is maintained unaltered in the reduced space. For each random reshuffling (default is 1000), a value from a metric is computed. The collection of all these values is used to draw the null model distribution. This distribution is employed to compute the probability to get at random a separation equal or higher than the one detected by using the original labels.

This program will compute a trustworthiness pvalue given an observable
value (xObs) from a metric obtained a-priori, the labels of a dataset
and their respective values (also known as scores).
 
The metrics available are:
    
a. Mann-Witheny Pvalue ('MWpval')
b. Permutation test by mean ('permmean')
c. Permutation test by median ('permmedian')
d. Area under the ROC-Curve ('AUC')
e. Area under the Precision-Recall curve ('AUPR')
f. Matthews Correlation Coefficient ('MCC')
g. F-score ('Fscore')
h. Accuracy ('accuracy')
i. Pearson correlation('pearscorr')
j. Spearman correlation('spearcorr')
    
Inputs

-labels: integer numeric vector with the sample label information.
example: [1 1 1 1 2 2 2 1 2 2 2 1 2 2 1]
    
-labelValues: floating numeric vector with sample scores. The ordering
should correspond with the labels variable.
example: [0.12 0.42 0.26 0.64 0.75 0.83 0.23 0.75 0.74 0.45 0.44 0.87 0.54 0.60 0.09]
    
-metric: string referring to the metric for which to compute the
trustworthiness.
example: 'AUC'
    
-xObs: observable computed value from the respective metric
example: 0.62 (calculated as the AUC of the example labels and labelValues)
    
Optional inputs:

-iters: Numeric value for iterations to create the null model.
Default: 1000

-plotFlag: binary value for the generation of the distribution plot after 
trustworthiness calculation
Default: true
    
Outputs:

-trustworthiness: trustworthiness pvalue calculated from the null model
taking as reference the observable input value xObs.

-xRand: null model with all the metric performances from the resampling 
approach. Its length is the same as the iters variable value.
    
If used, please reference:
    (Article accepted. Reference will be updated after publication).    
