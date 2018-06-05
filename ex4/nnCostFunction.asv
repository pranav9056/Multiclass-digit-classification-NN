function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1) X];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for i = 1:m
    x = X(i,:);
    [hQ,a2,z3,z2] = feedForward(Theta1,Theta2,x);
    % setting the Y vector
    Y = zeros(num_labels,1);
    Y(y(i)) = 1;
    
    % back prop
    delta3 = hQ' - Y;
    delta2 = (Theta2' * delta3) .* [0;sigmoidGradient(z2')];
    delta2 = delta2(2:end);
    Theta2_grad = Theta2_grad + delta3 * a2;
    %delta1 = (Theta1' * delta2(2:end)) .* sigmoidGradient(x');
    Theta1_grad = Theta1_grad + delta2 * x;
    % cost function calculation
    temp = -1* (Y .* log(hQ') + (1-Y) .* log(1-hQ'));
    J = J + sum(temp);
   
end

Theta1_grad = Theta1_grad ./ m + (lambda/m)* [zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad ./ m + (lambda/m)* [zeros(size(Theta2,1),1) Theta2(:,2:end)];


J = J/m;
RegularizedJ = sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2));
RegularizedJ = lambda*RegularizedJ/(2*m);
J = J + RegularizedJ; 

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
