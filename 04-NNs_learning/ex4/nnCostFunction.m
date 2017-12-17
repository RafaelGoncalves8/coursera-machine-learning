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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% y = mx1
% X = mxn
% a3 = oxm
% Theta1 = (k+1)x(n+1)
% Theta2 = ox(k+1)
% n = input layer size
% k = hidden layer size
% o = output layer size

yv=[1:num_labels] == y;

g = @(x) sigmoid(x);
g_prime = @(z) sigmoidGradient(z);

% Forwardpropagation (feedforward)
a1 = [ones(size(X,1),1), X]; % 5000x401
z2 = a1*Theta1';
a2 = [ones(size(z2,1),1), g(z2)]; %5000x26
z3 = a2*Theta2'; % 5000x10
a3 = g(z3);

h = a3;

% Non-regularized cost function
J += (sum(sum((-yv).*log(h) - ...
    (ones(size(yv))-yv).*log(ones(size(h)) - h))))/m;

% Regularized cost function
J += lambda*(sum(sum((Theta1(:,2:size(Theta1,2)).^2))) + ...
             sum(sum((Theta2(:,2:size(Theta2,2)).^2))))/(2*m);

% Backpropagation
d3 = h - yv;
Delta2 = a2'*d3;
Theta2_grad = Delta2/m;
% til here backprop is working

size(d3)
% d2 = Theta2(:,2:end)'*(d3 .* g_prime(z2)); % 25x5000 .* 10x5000 (?)
d2 = d3*Theta2(:,2:end) .* g_prime(z2); % 25x5000 .* 10x5000 (?)
Delta1 = d2*a1;
Theta1_grad = Delta1/m;

% Regularized backpropagation
Theta2_grad(:,2:end) += lambda*Theta2(:,2:end)/m;
Theta1_grad(:,2:end) += lambda*Theta1(:,2:end)/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
