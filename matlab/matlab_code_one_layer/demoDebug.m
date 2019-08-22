%% Initialization
clear; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 15;
hidden_layer_size = 300;
num_labels = 3;

% Load Training Data
fprintf('\nLoading training data ...\n');
dat1 = csvread('../SensorData/log_motegi_2laps_fwd.csv');
dat2 = csvread('../SensorData/log_motegi_2laps_bck.csv');
sensor_data = [dat1; dat2];

[X, y_raw] = preprocessData(sensor_data);
y = transformInput(y_raw);
m = size(X, 1);

%% Initializing Pameters
fprintf('\nInitializing Neural Network Parameters ...\n');

initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_theta1(:) ; initial_theta2(:)];

%% Training NN
fprintf('\nTraining Neural Network... \n');

lambda = 1.0;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

options = optimset('MaxIter', 1000);
tic;
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
toc;

theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% Predict
pred = predict(theta1, theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


% fprintf('\nLoading testing data ...\n');
% sensor_data_test = csvread('../SensorData/log_motegi_2laps_bck.csv');
% [X_test, y_raw_test] = preprocessData(sensor_data_test);
% y_test = transformInput(y_raw_test);
% 
% pred_test = predict(theta1, theta2, X_test);
% fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);

%% Write parameters
csvwrite('./parameters/theta1.csv', theta1);
csvwrite('./parameters/theta2.csv', theta2);
