function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);

% feedforward, izracunaj h_theta(x)
a1 = [ones(1,m); X'];
z2 = Theta1 * a1;
a2 = [ones(1,m); sigmoid(z2)];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

[~, p] = max(a3, [], 1);
p = p';

end
