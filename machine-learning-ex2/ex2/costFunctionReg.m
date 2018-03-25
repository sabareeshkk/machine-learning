function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

z = X * theta;
Htheta1 = log(sigmoid(z));
Htheta0 = log(1 - sigmoid(z));
mConst = 1/m;

% removing theta0 because we are not regularizing theta zero
% because the theta0 not inclided to regularization because x0=1 always
regTheta = [0;theta(2:end, :);];

% costfunction 
% have to take y' for multiplication
J = (((y' * Htheta1) + ((1 - y)' * Htheta0)) * -mConst) + (lambda/ (2 * m)) * sum(regTheta .^ 2);

HthetaMinusy = sigmoid(z) - y;
% HthetaMinusy' matrix multiplication
% gradient
grad = ((X' * HthetaMinusy) *  mConst) + (lambda/m) * regTheta;




% =============================================================

end
