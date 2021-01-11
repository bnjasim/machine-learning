function W = least_squares(A, Y)
    
    % number of data points
    m = size(A, 1);

    % Design matrix; append a column of ones before A
    X = [ones(m,1), A];

    % Least Squares Regression has a closed form solution
    W = pinv(transpose(X)*X) * transpose(X) * Y;

end