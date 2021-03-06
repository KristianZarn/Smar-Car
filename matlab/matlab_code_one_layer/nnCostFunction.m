function [J, grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2
theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

m = size(X, 1);

% pretvori oznake y v vektorje
vec_labels = eye(num_labels);
y_mat = vec_labels(:,y);

% feedforward, izracunaj h_theta(x)
a1 = [ones(1,m); X'];
z2 = theta1 * a1;
a2 = [ones(1,m); sigmoid(z2)];
z3 = theta2 * a2;
a3 = sigmoid(z3);

% izracunaj cost function
J = (1/m) * sum(sum((-y_mat .* log(a3) - (1-y_mat) .* log(1-a3))));

% cost funkciji dodaj regularizacijo
J = J + (lambda/(2*m)) * ...
    (sum(sum(theta1(:,2:end).^2)) + sum(sum(theta2(:,2:end).^2)));

% backpropagation
d3 = a3 - y_mat;
d2 = (theta2(:,2:end)' * d3) .* sigmoidGradient(z2);

theta1_grad = d2 * a1';
theta2_grad = d3 * a2';

% normalization and regularization
theta1_grad = theta1_grad / m;
theta2_grad = theta2_grad / m;

theta1_grad(:,2:end) = theta1_grad(:,2:end) + ...
    (lambda/m) .* theta1(:,2:end);
theta2_grad(:,2:end) = theta2_grad(:,2:end) + ...
    (lambda/m) .* theta2(:,2:end);

% Unroll gradients
grad = [theta1_grad(:) ; theta2_grad(:)];

end
