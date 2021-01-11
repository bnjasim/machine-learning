function pred = predict_neuralnet(net, data)
    % data should be n X n_input
    % Forward Pass
    n_data = size(data, 1);
    input = [ones(n_data, 1) data];
    Z1 = input * net.Theta1;
    % ReLU activation function
    A1 = [ones(n_data, 1) poslin(Z1)];
    Z2 = A1 * net.Theta2;
    % ReLU activation function
    A2 = [ones(n_data, 1), poslin(Z2)];
    Z3 = A2 * net.Theta3;
    % logistic sigmoid activation function
    pred = logsig(Z3); % y will be a vector of predictions

end