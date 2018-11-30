clc;
close all;
clear all;
num_epoch = 100;
classes = 10;
layers = [32*32, 800, classes];
learning_rate = 0.01;
train_accuracy=[];
valid_accuracy=[];
train_losses=[];
valid_losses=[];



load('../data/nist36_train.mat', 'train_data', 'train_labels')
load('../data/nist36_test.mat', 'test_data', 'test_labels')
load('../data/nist36_valid.mat', 'valid_data', 'valid_labels')

[W1, b1] = InitializeNetwork(layers);
%Loading pre-trained weights and biases
load('../data/nist26_model_60iters.mat');
W{2}= [W{2};W1{2}];
b{2} = [b{2};b1{2}];
for j = 1:num_epoch
    
    %Shuffling the data
    ind = randperm(size(train_data,1));
    data = train_data(ind,:);
    labels = train_labels(ind,:);
    
    %Training the model
    [W, b] = Train(W, b, data, labels, learning_rate);

    %Testing the model on train data
    [train_acc, train_loss] = ComputeAccuracyAndLoss(W, b, train_data, train_labels);
    %Testing the model on validation data
    [valid_acc, valid_loss] = ComputeAccuracyAndLoss(W, b, valid_data, valid_labels);
    
    train_accuracy = [train_accuracy,train_acc];
    valid_accuracy = [valid_accuracy,valid_acc];
    train_losses = [train_losses,train_loss];
    valid_losses = [valid_losses,valid_loss];

    fprintf('Epoch %d - accuracy: %.5f, %.5f \t loss: %.5f, %.5f \n', j, train_acc, valid_acc, train_loss, valid_loss)
    if j ==1 
            
            valid_p = valid_loss;
        else
            if valid_loss < valid_p 
               
                best_valid_loss = valid_loss;
                best_valid_accuracy = valid_acc;
               
                best_w = W;
                best_b = b;
                valid_p = valid_loss ;
            end
     end
end
figure;
plot(1:num_epoch,train_losses,'LineWidth',1.5);
hold on;
plot(1:num_epoch,valid_losses,'LineWidth',1.5);
legend('Training Loss','Validation Loss');
title({'Cross Entrophy loss plot','Fine Tuning'});
xlabel('Number of epochs');
ylabel('Cross Entrophy Loss');



figure;
plot(1:num_epoch,train_accuracy,'LineWidth',1.5);
hold on;
plot(1:num_epoch,valid_accuracy,'LineWidth',1.5);
legend('Training Accuracy','Validation Accuracy');
title({'Accuracy plot','Fine Tuning'});
xlabel('Number of epochs');
ylabel('Accuracy');

save('nist26_model_finetuning_100.mat', 'W', 'b','best_w','best_b')
