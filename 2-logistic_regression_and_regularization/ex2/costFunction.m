function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

n = size(X,2);

h = @(x) sigmoid(x*theta);

%        mx1   mx1           mx1                 mx1
J = sum(-y.*log(h(X)) - (ones(m,1)-y).*log(ones(m,1) - h(X)))/m;

for j = 1:size(theta,1),
    %               1xm       mx1
    grad(j) = sum((h(X) - y)'*X(:,j))/m;
end;

% =============================================================

end
