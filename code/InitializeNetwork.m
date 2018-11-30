function [W, b] = InitializeNetwork(layers)
% InitializeNetwork([INPUT, HIDDEN, OUTPUT]) initializes the weights and biases
% for a fully connected neural network with input data size INPUT, output data
% size OUTPUT, and HIDDEN number of hidden units.
% It should return the cell arrays 'W' and 'b' which contain the randomly
% initialized weights and biases for this neural network.

% Your code here

    %Initializing cell format
    W={};
    b={};
    
    % hidden layers
    num_layers = length(layers)-1;

    % Hidden layer of shape hiddenxinput format
    mean = 0;
    for num = 1:num_layers
        
        sigma = sqrt(2/(layers(num)+layers(num+1)));
        W{num} = normrnd(mean,sigma,[layers(num+1),layers(num)]);
        b{num} = zeros([layers(num+1),1]);

    end
    C = size(b{end},1);
    assert(size(W{1},2) == 1024, 'W{1} must be of size [H,N]');
    assert(size(b{1},2) == 1, 'b{end} must be of size [H,1]');
    assert(size(W{end},1) == C, 'W{end} must be of size [C,H]');

end
