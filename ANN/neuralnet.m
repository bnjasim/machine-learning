% Declare and initialize a 3 layer neural network
% Regression neural network, hence n_output = 1
classdef neuralnet
   properties
      Theta1
      Theta2
      Theta3
   end
   methods
      function obj = neuralnet(n_input, n_h1, n_h2)
         if nargin == 3
            epsilon_init = 0.12;
            obj.Theta1 = rand(n_input + 1, n_h1) * 2 * epsilon_init - epsilon_init;
            obj.Theta2 = rand(n_h1 + 1, n_h2) * 2 * epsilon_init - epsilon_init;
            obj.Theta3 = rand(n_h2 + 1, 1) * 2 * epsilon_init - epsilon_init;
         end
      end
   end
end