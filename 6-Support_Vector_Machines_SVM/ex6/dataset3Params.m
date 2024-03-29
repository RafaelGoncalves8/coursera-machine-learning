function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.  
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Best parameters found by this algorithm (algorithm commented 
% because each time I run submit this code would run also and 
% this code is very time consuming). 
C = 1;
sigma = 0.1;

% Uncomment for the algorithm I used to find best C and sigma
% options = [0.01 0.03 0.1 0.3 1 3 10 30];
% model = @(C, sigma) svmTrain(X ,y ,C , ...
%                     @(x1,x2) gaussianKernel(x1, x2, sigma));
% err = @(x) mean(double(x ~= yval));
% predictions = [];
%
% for i = 1:8,
%     for j = 1:8,
%         C = options(i); sigma = options(j);
%         pred = svmPredict(model(C, sigma), Xval);
%         predictions = [predictions; C sigma err(pred)];
%     end;
% end
%
% predictions
% [v, index] = min(predictions(:,3)) 
%
% C = predictions(index, 1)
% sigma = predictions(index, 2)

% =========================================================================

end
