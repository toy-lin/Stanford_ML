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

# Part 1 -----------------------------------
# forward propagation
input = [ones(m,1) X];
hidden = sigmoid(input * Theta1');
hidden = [ones(m,1) hidden];
output = sigmoid(hidden * Theta2');

# map y to binary vector, i can't find a more efficient way
temp_y = zeros(m,num_labels);
for i = 1:m,
  temp_y(i,y(i,1)) = 1;
endfor

# calculate regularization exclude bias term
temp_t1 = Theta1;
temp_t2 = Theta2;
temp_t1(:,1) = 0;
temp_t2(:,1) = 0;
reg = lambda/(2*m) * (sum((temp_t1.^2)(:)) + sum((temp_t2.^2)(:)));

# calculate cost
J = sum(sum(-temp_y.*log(output)-(1-temp_y).*log(1-output),2))/m + reg;

# Part 2 -----------------------------------
# backward propagation
e3 = output .- temp_y; #(m,num_labels)
e2 = e3 * Theta2; #(m,hidden_layer_size+1)

# remove bias term
e2 = e2(:,2:hidden_layer_size+1);#(m,hidden_layer_size)
# real e2
temp_hidden = hidden(:,2:hidden_layer_size+1);
e2 = e2 .* (temp_hidden .* (1 .- temp_hidden));#(m,hidden_layer_size)

Theta2_grad = (hidden' * e3)'./m; #(hidden_layer_size+1,m)*(m,num_labels)
Theta1_grad = (input' * e2)'./m; #(input_layer_size+1,m)*(m,hidden_layer_size)

# Part 3 -----------------------------------

Theta1_grad = Theta1_grad .+ (lambda .* temp_t1)/m;
Theta2_grad = Theta2_grad .+ (lambda .* temp_t2)/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
