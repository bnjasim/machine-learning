## Chow Liu Algorithm
It is an extension of the Naive Bayes classifier. Naive Bayes has an independence assumption of features. In Chow Liu features have a first order dependence over each other. The class conditional probability of a sample over its features is calculated by factoring over a tree structure of the features. 

P (C = c|X1, . . . , X36) ∝ P(X1, . . . , X36|C=c) * P(C=c)
The prior probability P(C=c) is estimated from the training data.
The class conditional probabilities P(X1, . . . , X36|C=c) are computed using the learned trees T1 & T2 for class 1 & class 2 respectively.

Then choose the predicted class as 1 if P(C=1| X1, . . . , X36 ) > P(C=2| X1, . . . , X36 ) Otherwise choose the predicted class as 2.

The algorithm expects the train and test data in numerical format. Each features should take values from 1,2,3 etc. Classes should also be numerical 1,2 etc.


    % Train
    [prior, P_X, P_XY, mst] = train_Chow_Liu(train_data, train_labels, fmax, nclass);


    % Test on the train data
    accuracy = test_Chow_Liu(train_data, train_labels, nclass, prior, P_X, P_XY, mst);

Prediction Accuracy on the train data = 94.24%
Prediction Accuracy on the test data = 93.25%

