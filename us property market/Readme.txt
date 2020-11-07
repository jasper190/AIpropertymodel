============================================
a. Project.
============================================
Model development for the US market

============================================
b. Overview of the submitted folder and the folder structure.
============================================
```
├── src
│ ├── module1.py
│ └── module2.py
├── data
│ └── home_sales.db
├── README.md
├── eda.ipynb
├── config.csv
├── requirements.txt
└── run.sh


============================================
c. Instructions for executing the pipeline and modifying any parameters.
============================================

.sh file- Replace parameters [ -a "path to data" -b "path to configfile" ] with the actual paths and run the script

configfile - A csv file with parameters that will be read by pandas into the system.

============================================
d. Description of logical steps/flow of the pipeline. If you find it useful, please feel
free to include suitable visualization aids (eg, flow charts) within the README.
============================================

[read_config] -> [Queries SQL] -> [Cleans dataframe from nulls and duplicates] -> [based on target variable type, classifies or regresses] -> outputs list of models which it returns.


============================================
e. Overview of key findings from the EDA conducted in Task 1 and the choices
made in the pipeline based on these findings, particularly any feature
engineering. Please keep the details of the EDA in the `.ipynb`, this section should
be a quick summary.
============================================

Initial observations when cleaning:
1) there are duplicate transaction ids, such as ( 7259748, 7964035, 6737999) which appear multiple times
2) dates will require some cleaning , values such as '11 Auust 2014' appear
3) 7 or more bedroom units are quite rare, might want to group them together, might appear as outliers 
4) Number of bathrooms - have decimal places, consider to drop this column
5) condition - needs stripping and standardization of text

More than the 75 percentile of houses did not undergo renovation, or was underreported, so the proportion 
of houses that had renovated at all is a small proportion.

The age range of houses is quire wide, ranging from -1 to 115, and a median of 40 years.

<Correlation check>
1. Number of floors and the year the property age seems to have a strong correlation ratio of -0.6.
2. Living room size and price looks to have a strong correlation ratio of 0.6.

(living room size, no of bedrooms) - moderate correlation of 0.4
(price/sqm, no of floors) - moderate correlation of 0.4
(price, view) - moderate correlation of 0.4
(longtitude , year built) - moderate correlation of 0.4

I used PermutationImportance to permutate the various features to get a sense of important features. Before standardizing features by removing the mean and scaling to unit variance.To get a sense of a good number of variables to use for the model, I used PCA to
discover that first 12 components are likely to be sufficient for the model to span the dataset space.

For the classification task, I broke the price target variable into segments of 200k to have a sense of the price, because the first 200k would roughly capture up to approximately
the first decile. and the max value after removal of the 99th percentile and above is slighltly less than 2 mil. That would 
give 10 segments. So that the output estimate is more meaningful as compared to using the percentiles where between the
20th and 80th percentile the jumps are not very significant.

Added 11th group as those above 2mil to keep the information in the model. ( Max value is 7.7mil)


============================================
f. Explanation of your choice of models for each machine learning task.
g. Evaluation of the models developed. Any metrics used in the evaluation should
also be explained.
============================================

<Regression>
In terms of choosing a regression model, one factor to consider is we want to fit a global function, like a linear
regression, or fit a model with more localized decision boundaries, eg a random forest model. 

A linear regression model would allow for simpler global trend explainations, whereas for tree based models like random 
forest which utilizes bagging.

An alternative tree method that could have been considered would be boosting, where each weak learner learns from the
previous iteration. But would be harder to tune correctly.

In terms of random forest cons, the model is prone to overfitting, which may lead to problems when generalizing future
test datasets.

In terms of measuring, I chose 2 metrics, explained_variance, and R2 (or goodness of fit), taking the average of scores
that were determined by K-folds, where K-1 folds formed the training set and 1 fold formed the test set to help keep the
random picking process.

explained_variance - this metric tells me an estimate of how much of the variation in the dataset is currently explained/
accounted for.

R2 - This is a goodness of fit metric that indicates the percentage of the variance in 
the dependent variable that the independent variables explain collectively.

<Classification>
Main models used were, listed with their accuracy scores.

LogReg  :  0.5040629761300153
SVM  :  0.9846368715083799
DecTree  :  1.0
KNN  :  0.9991112239715592
GaussianNB  :  0.9385474860335196

DecTree is probably overfitting the dataset. Similarly, how flexible the decision boundary depends on the choice of model as well
as the choice of hyper tuning parameter. For example for KNN, the main tuning parameter is the number of neighbours, and the algo
works by finding the aggregate of the K nearest neighbours, to decide and classify a given point. So a high value of K would
make the decision boundary less flexible and vice versa.


In terms of model choice, I would go with the random forest model, as even with k folds cross validation, the accuracy was on average
above 80% .

However , giving bands of acceptable price ranges to property agents would be easier for them to self serve.


============================================
h. Other considerations for deploying the models developed.
============================================

If I had more time, I would like to improve especially the classification models by introducing ROC curves to measure the model's viability as well as use the various models AUC for comparabiity. only choosing models with AUC above 0.5.

An alternative type of classification I could have tried was a one vs rest model, where the model guesses in stages if the price was part of a paticular band or not, for each of the 11 bands. That would probably have made it easier and more feasible to implement such measures.
