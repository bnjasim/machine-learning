% Variational Inference for Mixture of Univariate Gaussians
clc
clear all

% Load data
data = load('data2.txt');

N = length(data); % Number of datapoints
K = 5; % Number of clusters

% Initialize parameters
alpha_0 = ones(1, K); % Dirichlet prior 
beta_0 = ones(1, K); % Gaussian mean precision scaling
rng(2); % seed for random number generation; fix it for reporducibility
m_0 = rand(1, K); % initial means of the mixture components (clusters)
a_0 = ones(1, K); % Gamma distribution; first parameter
b_0 = ones(1, K); % Gamma distribution; second parameter

% soft cluster assignment of each data point
% probability that nth data point belongs to kth cluster
r_nk = zeros(N, K);
x_n = repmat(data, [1,K]);

% copy parameters
alpha_k = alpha_0;
beta_k = beta_0;
m_k = m_0;
a_k = a_0;
b_k = b_0;

tol = 1e-10; % tolerance for convergence
N_k = zeros(1, K); % total contribution of each cluster
N_p = ones(1, K); % previous value of N_k
epoch = 0;

while(norm(N_k - N_p) > tol)
    epoch = epoch + 1;
    fprintf(1, "Epoch %d\n", epoch);
    
    % E-Step equivalent: compute soft cluster assignment of each data point
    for n = 1:N
        xn = data(n);
        ln_pi = psi(alpha_k) - psi(sum(alpha_k));
        ln_phi = psi(a_k) - log(b_k);
        residual = (xn - m_k) .* (xn - m_k);
        E_mu_phi = (1 ./ beta_k) + residual .* a_k ./ b_k;
        rho = exp(ln_pi + 0.5 * ln_phi - 0.5 * E_mu_phi); % 1 x K vector
        r_nk(n, :) = rho / sum(rho); % Normalize rho to get probability values
    end

    N_p = N_k; % keep the previous N_k value

    N_k = sum(r_nk);
    x_k = sum(r_nk .* data) ./ N_k;
    S_k = sum(r_nk .* (x_n - x_k) .* (x_n - x_k)) ./ N_k;


    % M-Step equivalent: Update parameters
    % Update
    alpha_k = alpha_0 + N_k;
    beta_k = beta_0 + N_k;
    m_k = (beta_0 .* m_0 + N_k .* x_k) ./ beta_k;
    a_k = a_0 + N_k / 2;
    b_k = b_0 + 0.5 * (N_k .* S_k + (beta_0 .* N_k ./ beta_k) .* (x_k - m_0) .* (x_k - m_0));

end
disp("Algorithm Converged!")
disp(N_k);
