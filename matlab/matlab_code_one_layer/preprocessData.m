function [X, y] = preprocessData(sensor_data)
%% Keep all data
valid = ~(sensor_data(:,19) == 1 & sensor_data(:,20) == 1);
sensor_data = sensor_data(valid, :);
X = sensor_data(:,3:17);
y = sensor_data(:,19:20);

%% Filter data
% fwd = sensor_data(:,18) == 1 & sensor_data(:, 19) == 0 & sensor_data(:, 20) == 0;
% left = sensor_data(:,18) == 1 & sensor_data(:, 19) == 1 & sensor_data(:, 20) == 0;
% right = sensor_data(:,18) == 1 & sensor_data(:, 19) == 0 & sensor_data(:, 20) == 1;
% 
% tmp = find(fwd);
% discard_num = sum(fwd) - round(mean([sum(left), sum(right)]));
% discard = randperm(length(tmp), discard_num);
% fwd(tmp(discard)) = 0;
% 
% valid = fwd | left | right;
% sensor_data = sensor_data(valid, :);
% X = sensor_data(:,3:17);
% y = sensor_data(:,19:20);

end