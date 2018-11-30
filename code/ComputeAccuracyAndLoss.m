function [accuracy, loss] = ComputeAccuracyAndLoss(W, b, data, labels)
% [accuracy, loss] = ComputeAccuracyAndLoss(W, b, X, Y) computes the networks
% classification accuracy and cross entropy loss with respect to the data samples
% and ground truth labels provided in 'data' and labels'. The function should return
% the overall accuracy and the average cross-entropy loss.
[~,N] = size(data);
[~,C] = size(labels);
assert(size(W{1},2) == N, 'W{1} must be of size [~,N]');
assert(size(b{1},2) == 1, 'b{1} must be of size [~,1]');
assert(size(b{end},2) == 1, 'b{end} must be of size [~,1]');
assert(size(W{end},1) == C, 'W{end} must be of size [C,~]');

%Your code here
    %Classification....
    [outputs] = Classify(W,b,data);
    %Loss....
    e = labels.*log(outputs) ;
    e = sum(e(:));
    e=e./size(labels,1);
    loss = -e;
    
    %Accuracy....
    [~,i]=max(labels');
    [~,j]=max(outputs');
    count = sum(i==j);
    accuracy=count./size(labels,1);

end
