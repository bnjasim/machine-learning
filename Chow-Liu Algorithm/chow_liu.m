% Chess Winner Prediction using Chow Liu Algorithm
clc
clear all

% Rename kr-vs-kp.data file to kr-vs-kp.csv 
% Load the data
mytable = readtable('kr-vs-kp.csv','ReadVariableNames', false);
mydata = table2array(mytable);
[m,n] = size(mydata);

% Convert mydata to a numeric matrix
% Need to convert character features (f or t) to integers (1 or 2 or 3)
% The classes are won = 1, nowin = 2 

firstRow = mydata(1,:);
% initialize a matrix with all values = 2 and change to 1 if 
% features match with the first row.
% Feature 15 can take 3 values. (n=1, w=2, b=3)
data = 2 * ones(m, n);

for row = 1:m
    for col = 1:n
        if strcmp(mydata{row, col}, firstRow{col})
            data(row, col) = 1;
        end

        if (col == 15) && strcmp(mydata{row, col}, 'b')
            data(row, col) = 3;
        end
    end
end

% Maximum value each feature can take. e.g. fmax(15) should be 3
fmax = max(data);
fmax = fmax(1:end-1); % last column is the class label; remove it.

% Number of classes
nclass = 2;

% Randomly shuffle data 
rng(2); % random seed to ensure reproducibility
idx = randperm(m);
data = data(idx, :);

% Train & Test split
train_data = data(1:2500, 1:end-1);
train_labels = data(1:2500, end);

test_data = data(2501:end, 1:end-1);
test_labels = data(2501:end, end);

% Train
[prior, P_X, P_XY, mst] = train_Chow_Liu(train_data, train_labels, fmax, nclass);


% Test on the train data
accuracy = test_Chow_Liu(train_data, train_labels, nclass, prior, P_X, P_XY, mst);
fprintf(1, 'accuracy on training data = %.2f %% \n', accuracy);

% Test on the test data
accuracy = test_Chow_Liu(test_data, test_labels, nclass, prior, P_X, P_XY, mst);
fprintf(1, 'accuracy on test data = %.2f %% \n', accuracy);
