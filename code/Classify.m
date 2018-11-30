function [outputs] = Classify(W, b, data)
% [predictions] = Classify(W, b, data) should accept the network parameters 'W'
% and 'b' as well as an DxN matrix of data sample, where D is the number of
% data samples, and N is the dimensionality of the input data. This function
% should return a vector of size DxC of network softmax output probabilities.
[D,N] = size(data);
C = size(b{end},1);
assert(size(W{1},2) == N, 'W{1} must be of size [H,N]');
assert(size(b{1},2) == 1, 'W{end} must be of size [H,1]');
assert(size(W{end},1) == C, 'W{end} must be of size [C,H]');

% Your code here
for sample_num = 1:size(data,1)
   [outputs(sample_num,:),~,~]=Forward(W, b, data(sample_num,:)');     
end

assert(all(size(outputs) == [D,C]), 'output must be of size [D,C]');
end
