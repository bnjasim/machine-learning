% Residual Sum of Squares
function r = RSS(A, Y)
    
    % number of data points
    m = size(A, 1);

    % Find least squares regression coefficients
    W = least_squares(A, Y);
    
    % Design matrix; append a column of ones before A
    X = [ones(m,1), A];
    
    % Find the predictions
    Pred = X * W;

    diff = Pred - Y;
    r = transpose(diff) * diff;

end