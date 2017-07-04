%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10; 

load('ex3data1.mat');
m = size(X, 1);


%% ================ Part 2: Loading Pameters ================
% Load the weights into variables Theta1 and Theta2
load('ex3weights.mat');

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


z_3 = [ones(m, 1) sigmoid([ones(m, 1) X] * Theta1')] * Theta2';
h_of_x = sigmoid(z_3);

[max_value, p] = max(h_of_x, [], 2);
p

%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end