function net = train_neuralnet(net, data, labels, n_epochs, lr, lambda)
    
    % n_epochs = 1000;
    % lr=0.5;
    % lambda = 1;

    disp('Training!!!');

    for epoch = 1:n_epochs

    total_loss = 0;

    for i = 1:length(labels)
        input = [1, data(i, :)]; % dim: 1 x (n_input+1)
        label = labels(i); 

        % Forward Pass
        Z1 = input * net.Theta1;
        % ReLU activation function
        A1 = [1, poslin(Z1)];
        Z2 = A1 * net.Theta2;
        % ReLU activation function
        A2 = [1, poslin(Z2)];
        Z3 = A2 * net.Theta3;
        % logistic sigmoid activation function
        y = logsig(Z3); % y will be a scalar

        % Back propagation
        delta = (y-label) * y * (1-y);
        A2d = zeros(size(A2));
        A2d(A2>0) = 1;
        delta3 = net.Theta3 * delta .* transpose(A2d);
        A1d = zeros(size(A1));
        A1d(A1>0) = 1;
        delta2 = net.Theta2 * delta3(2:end) .* transpose(A1d);

        % Update the weights
        net.Theta3 = net.Theta3 - lr * (delta * transpose(A2) + lambda * [0; delta(2:end)]);
        net.Theta2 = net.Theta2 - lr * (transpose(A1) * transpose(delta3(2:end)) + lambda * [zeros(1, size(net.Theta2, 2)); net.Theta2(2:end,:)]);
        net.Theta1 = net.Theta1 - lr * (transpose(input) * transpose(delta2(2:end)) + lambda * [zeros(1, size(net.Theta1, 2)); net.Theta1(2:end,:)]);

        loss = 0.5 * (y - label)^2;
        total_loss = total_loss + loss;

    end

    fprintf(1, 'Epoch %d Loss = %.2f\n', [epoch, total_loss]);

    end

end