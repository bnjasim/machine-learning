% cd ~/Desktop/Upwork/Final_Project_ML/DataSet/DataSet-Q3/
clear
clc
close all

% Read the dataset
mytable = readtable('CASE1201.ASC.txt');
% State name is an unnecessary column
data = mytable{:, 2:end};

A = data(:, 2:end);
Y = data(:, 1);

% number of data points
m = size(A, 1);
% number of features
n = size(A, 2);

% Least Squares Minimization
W_lsq = least_squares(A, Y);

% Stepwise selection
indices = stepwise_selection(A, Y);

% Lasso
lambda = 0.1 % Try 1, 5, 10, 20, 50, 100, 500, 1000, 5000
% Design matrix; append a column of ones before A
D = [ones(m,1), A];

% It is advisable to normalize the data to speed up convergence of lasso
max_values = max(D);
X = D ./ max_values;
W_lasso = lasso(X, Y, lambda);


% 2. Cross Validation (K=4)
% Randomly shuffle data X
idx = randperm(m);
X_p = X(idx, :);
A_p = A(idx, :);
Y_p = Y(idx);

% create 4 subsets of data with size 13, 13, 12 and 12
X1 = X_p(1:13, :);
A1 = A_p(1:13, :);
Y1 = Y_p(1:13);

X2 = X_p(14:26, :);
A2 = A_p(14:26, :);
Y2 = Y_p(14:26);

X3 = X_p(27:38, :);
A3 = A_p(27:38, :);
Y3 = Y_p(27:38);

X4 = X_p(39:50, :);
A4 = A_p(39:50, :);
Y4 = Y_p(39:50);

% Test on first subset and train on the rest

% 1. least_squares
W_lsq = least_squares([A2;A3;A4], [Y2;Y3;Y4]);
% Test it on the hold out subset
pred = [ones(size(Y1)), A1] * W_lsq;
diff = pred - Y1;
rss = transpose(diff)*diff;

% 2. stepwise_selection
indices = stepwise_selection([A2;A3;A4], [Y2;Y3;Y4]);
% All indices are selected, hence same as least squares

% 3. Lasso
lambda = 1;
W_lasso = lasso([X2;X3;X4], [Y2;Y3;Y4], lambda);
% Test it on the hold out subset
pred = X1 * W_lasso;
diff = pred - Y1;
rss = transpose(diff)*diff;


% Test on second subset and train on the rest

% 1. least_squares
W_lsq = least_squares([A1;A3;A4], [Y1;Y3;Y4]);
% Test it on the hold out subset
pred = [ones(size(Y2)), A2] * W_lsq;
diff = pred - Y2;
rss = transpose(diff)*diff;

% 2. stepwise_selection
indices = stepwise_selection([A1;A3;A4], [Y1;Y3;Y4]);
% Train a least_squares model with the selected indices
W_lsq = least_squares([A1(:, indices);A3(:, indices);A4(:, indices)], [Y1;Y3;Y4]);
% Test it on the hold out subset
pred = [ones(size(Y2)), A2(:, indices)] * W_lsq;
diff = pred - Y2;
rss = transpose(diff)*diff;

% 3. Lasso
lambda = 1;
W_lasso = lasso([X1;X3;X4], [Y1;Y3;Y4], lambda);
% Test it on the hold out subset
pred = X2 * W_lasso;
diff = pred - Y2;
rss = transpose(diff)*diff;



% Test on third subset and train on the rest

% 1. least_squares
W_lsq = least_squares([A1;A2;A4], [Y1;Y2;Y4]);
% Test it on the hold out subset
pred = [ones(size(Y3)), A3] * W_lsq;
diff = pred - Y3;
rss = transpose(diff)*diff;

% 2. stepwise_selection
indices = stepwise_selection([A1;A2;A4], [Y1;Y2;Y4]);
% Train a least_squares model with the selected indices
W_lsq = least_squares([A1(:, indices);A2(:, indices);A4(:, indices)], [Y1;Y2;Y4]);
% Test it on the hold out subset
pred = [ones(size(Y3)), A3(:, indices)] * W_lsq;
diff = pred - Y3;
rss = transpose(diff)*diff;

% 3. Lasso
lambda = 1;
W_lasso = lasso([X1;X2;X4], [Y1;Y2;Y4], lambda);
% Test it on the hold out subset
pred = X3 * W_lasso;
diff = pred - Y3;
rss = transpose(diff)*diff;



% Test on fourth subset and train on the rest

% 1. least_squares
W_lsq = least_squares([A1;A2;A3], [Y1;Y2;Y3]);
% Test it on the hold out subset
pred = [ones(size(Y4)), A4] * W_lsq;
diff = pred - Y4;
rss = transpose(diff)*diff;

% 2. stepwise_selection
indices = stepwise_selection([A1;A2;A3], [Y1;Y2;Y3]);
% Train a least_squares model with the selected indices
W_lsq = least_squares([A1(:, indices);A2(:, indices);A3(:, indices)], [Y1;Y2;Y3]);
% Test it on the hold out subset
pred = [ones(size(Y4)), A4(:, indices)] * W_lsq;
diff = pred - Y4;
rss = transpose(diff)*diff;

% 3. Lasso
lambda = 1;
W_lasso = lasso([X1;X2;X3], [Y1;Y2;Y3], lambda);
% Test it on the hold out subset
pred = X4 * W_lasso;
diff = pred - Y4;
rss = transpose(diff)*diff;








