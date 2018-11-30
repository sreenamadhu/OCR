function [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a)
% [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a) computes the gradient
% updates to the deep network parameters and returns them in cell arrays
% 'grad_W' and 'grad_b'. This function takes as input:
%   - 'W' and 'b' the network parameters
%   - 'X' and 'Y' the single input data sample and ground truth output vector,
%     of sizes Nx1 and Cx1 respectively
%   - 'act_h' and 'act_a' the network layer pre and post activations when forward
%     forward propogating the input smaple 'X'
N = size(X,1);
H = size(W{1},1);
C = size(b{end},1);
assert(size(W{1},2) == N, 'W{1} must be of size [H,N]');
assert(size(b{1},2) == 1, 'b{end} must be of size [H,1]');
assert(size(W{end},1) == C, 'W{end} must be of size [C,H]');
assert(all(size(act_a{1}) == [H,1]), 'act_a{1} must be of size [H,1]');
assert(all(size(act_h{end}) == [C,1]), 'act_h{end} must be of size [C,1]');

    %Initialization
    grad_L_g={};
    grad_W={};
    grad_b={};
    grad_L_a={};
    
    
    
    %Backpropagation
    for num = length(W):-1:1
        if num == length(W)
            %grad_L_a of shape Cx1
            grad_L_a{length(W)} = act_h{end} - Y;
            grad_W{num} = grad_L_a{num}*act_h{num-1}';
            grad_b{num} = grad_L_a{num};
        else
            grad_L_h{num} = W{num+1}'*grad_L_a{num+1};
            grad_L_a{num} = grad_L_h{num}.*(act_h{num}.*(1-act_h{num}));
            grad_b{num} = grad_L_a{num};
            if num >1
                grad_W{num} = grad_L_a{num}*act_h{num-1}';               
            else
                grad_W{num} = grad_L_a{num}*X';
            end
        end       
    end

% Your code here
assert(size(grad_W{1},2) == N, 'grad_W{1} must be of size [H,N]');
assert(size(grad_W{end},1) == C, 'grad_W{end} must be of size [C,N]');
assert(size(grad_b{1},1) == H, 'grad_b{1} must be of size [H,1]');
assert(size(grad_b{end},1) == C, 'grad_b{end} must be of size [C,1]');

end
