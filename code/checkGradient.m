clc;
close all;

%Data Loading
load('../data/nist26_train.mat');

% Initializing the hyper-parameters for the network
classes = 26;
layers = [32*32, 400, classes];
learning_rate = 0.001;

% Parameter initialization
[W, b] = InitializeNetwork(layers);

%Calculating the gradients of W and b in the network
[~, act_h, act_a] = Forward(W, b, train_data(1,:)');
[grad_W, grad_b] = Backward(W, b, train_data(1,:)', train_labels(1,:)', act_h, act_a);

%Gradient Checker for weights
episilon = 10^-4;

%Loss for original W and b
[~, loss] = ComputeAccuracyAndLoss(W, b, train_data, train_labels); 

%Number of layers
n = length(W);
%Loop through every gradient of W and b.
for num = 1:n
    
    for i = 1:size(W,1)
        for j = 1:size(W,2)
            
            %Creating a copy of the weight matrices on which checking is done
            W_lower = W;
            W_higher = W;
            
            %Vary the Weight vectors by episilon
            W_higher{num}(i,j) = W{num}(i,j) + episilon;
            W_lower{num}(i,j) = W{num}(i,j) - episilon;
            
            %Computing the loss by the updated weight vectors
            [~, loss_higher] = ComputeAccuracyAndLoss(W_higher, b, train_data(1,:), train_labels(1,:)); 
            [~, loss_lower] = ComputeAccuracyAndLoss(W_lower, b, train_data(1,:), train_labels(1,:));
            
            %grad_W(i,j) -> True gradient
            grad_measured = (loss_higher-loss_lower)./(2*episilon);
            
            %Change in the gradients
            delta = abs(grad_W{num}(i,j)-grad_measured);
            
            assert( delta < episilon , 'Gradient of W{%d} at (%d,%d) is incorrect. Please check and come back:)',num,i,j);
            
        end
    end
end
fprintf('Gradients for weight matrices are perfect! :)\n');

%Loop through every gradient of b.
for num = 1:n
    
    for i = 1:size(b,1)
        
        %Creating a copy of the weight matrices on which checking is done
        b_lower = b;
        b_higher = b;
        
        %Vary the bias vectors by episilon
        b_higher{num}(i,1) = b{num}(i,1) + episilon;
        b_lower{num}(i,1) = b{num}(i,1) - episilon;
        
        %Computing the loss by the updated bias vectors
        [~, loss_higher] = ComputeAccuracyAndLoss(W, b_higher, train_data(1,:), train_labels(1,:));
        [~, loss_lower] = ComputeAccuracyAndLoss(W, b_lower, train_data(1,:), train_labels(1,:));
        
        %grad_b{num}(i,1) -> True gradient
        grad_measured = (loss_higher-loss_lower)./(2*episilon);
        
        %Change in the gradients
        delta = abs(grad_b{num}(i,1)-grad_measured);
        
        assert( delta < episilon , 'Gradient of b{%d} at (%d) is incorrect. Please check and come back:)',num,i);
        
        
    end
end
fprintf('Gradients for bias matrices are perfect! :)\n');