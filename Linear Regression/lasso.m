% lasso implementation using proximal gradients method
function [Wk, iter] = lasso(X, Y, lambda)
    % number of data points
    m = size(X, 1);
    % number of covariates + intercept
    n = size(X, 2);

    % initialize parameters
    Wk = zeros(n, 1);

    % set the learning rate tau
    tau = 0.0045; % should be less than 1/||X||^2
    Lambda = lambda * ones(size(Wk));
    Lambda(1) = 0; % No shrinkage to the intercept term
    tol = 1e-6; % convergence tolerance value

    W_prev = Wk;
    iter = 0;
    while true
        grad = transpose(X) * (X * Wk - Y);
        Zk = Wk - tau * grad;

        % update Wk
        shrink = abs(Zk) - Lambda * tau/2;
        shrink(shrink < 0) = 0;
        Wk = sign(Zk) .* shrink;

        iter = iter + 1;
        if (norm(Wk - W_prev) < tol)
            disp(iter)
            break
        else
            W_prev = Wk;
        end
    end
end