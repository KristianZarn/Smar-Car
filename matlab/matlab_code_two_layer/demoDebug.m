%% Initialization
clear; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 15;
hidden_layer1_size = 15;
hidden_layer2_size = 10;
num_labels = 2;

% Load Training Data
fprintf('\nLoading training data ...\n');
dat1 = csvread('../SensorData/log_motegi_1_fwd.csv');
sensor_data = dat1;

[X, y_raw] = preprocessData(sensor_data);
y = transformInput(y_raw);
m = size(X, 1);

%% Initializing Pameters
fprintf('\nInitializing Neural Network Parameters ...\n');

initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_theta2 = randInitializeWeights(hidden_layer1_size, hidden_layer2_size);
initial_theta3 = randInitializeWeights(hidden_layer2_size, num_labels);
initial_nn_params = [initial_theta1(:) ; initial_theta2(:) ; initial_theta3(:)];

%% Training NN
fprintf('\nTraining Neural Network... \n');

lambda = 0.1;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   num_labels, X, y, lambda);

options = optimset('MaxIter', 1000);
tic;
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
toc;

% iterations = 10;
% alpha = 1.0;
% tic;
% [nn_params, cost] = gradientDescent(costFunction, initial_nn_params, iterations, alpha);
% toc;

bound1 = hidden_layer1_size * (input_layer_size + 1);
bound2 = bound1 + hidden_layer2_size * (hidden_layer1_size + 1);

theta1 = reshape(nn_params(1:bound1), ...
                 hidden_layer1_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + bound1):bound2), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));
             
theta3 = reshape(nn_params((1 + bound2):end), ...
                 num_labels, (hidden_layer2_size + 1));

%% Predict
pred = predict(theta1, theta2, theta3, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% fprintf('\nLoading testing data ...\n');
% sensor_data_test = csvread('../SensorData/log_motegi_1_fwd.csv');
% [X_test, y_raw_test] = preprocessData(sensor_data_test);
% y_test = transformInput(y_raw_test);
% 
% pred_test = predict(theta1, theta2, theta3, X_test);
% fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);

%% Write parameters
csvwrite('./parameters/theta1.csv', theta1);
csvwrite('./parameters/theta2.csv', theta2);
csvwrite('./parameters/theta3.csv', theta3);