function p = predict(theta1, theta2, theta3, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);

% feedforward, izracunaj h_theta(x)
a1 = [ones(1,m); X'];
z2 = theta1 * a1;
a2 = [ones(1,m); sigmoid(z2)];
z3 = theta2 * a2;
a3 = [ones(1,m); sigmoid(z3)];
z4 = theta3 * a3;
a4 = sigmoid(z4);

[~, p] = max(a4, [], 1);
p = p';

end
