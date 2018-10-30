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

values_try = [0.01 0.03 0.1 0.3 1.0 3.0 10 30];

error_min_index = [1 1];
error_min = intmax;
for i = 1 : length(values_try),
  C = values_try(i);
  for j = 1:length(values_try),
    sigma = values_try(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    y_output = svmPredict(model,Xval);
    error = mean(double(y_output~=yval));
    sprintf("C:%f sigma:%f error:%f\n",C,sigma,error)
    if error < error_min,
      error_min = error;
      error_min_index = [i j];
    endif
  endfor
endfor

C = values_try(error_min_index(1));
sigma = values_try(error_min_index(2));

% =========================================================================

end
