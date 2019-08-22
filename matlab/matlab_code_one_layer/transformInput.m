function label = transformInput(y)
label = zeros(size(y,1), 1);
label(y(:,1) == 0 & y(:,2) == 0) = 1;
label(y(:,1) == 0 & y(:,2) == 1) = 2;
label(y(:,1) == 1 & y(:,2) == 0) = 3;
end