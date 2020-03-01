%27/12/2019 Radial Basis Function Network with Center Selection using OLS
%(Orthogonal Least Squares) method. This algorithm generates centers and
%widths for RBFs and then orders RBFs from highest energy to lowest. After
%selecting RBFs with highest energy then it optimize weights and other
%parameters using Levenberg-Marquardt Method.
%Enes AKGUN, Pamukkale University
%Final project for Optimization Techniques
close all;
clear all;
clc;

% ------------------------------------------------
% load henondata x;
% Z = x;
% NumberOfInputs = 4;
% LengthOfTimeSeries =1500;
% PredictHorizon = 50;
% lambda = 0.0;
% ------------------------------------------------
% t = [-3000:1:3000];
% NumberOfInputs = 3;
% LengthOfTimeSeries = 3000;
% PredictHorizon = 100;
% yt = [sin(t)./t]; 
% x = yt + 0.01*randn(length(yt),1);
% Z=x;
% ------------------------------------------------

% load SunSpot Z;
% Z = Z';
% x = Z;
% NumberOfInputs = 100;
% LengthOfTimeSeries = 3000;
% PredictHorizon = 15;
% ------------------------------------------------

    data = readtable('DATA.xlsx');
    data = table2array(data);
    LengthOfTimeSeries =8400;
    PredictHorizon = 100;
    x = data(1:LengthOfTimeSeries,1:4);
    x_test = data(LengthOfTimeSeries+1:LengthOfTimeSeries+PredictHorizon,1:4);
    y = data(1:LengthOfTimeSeries,5);
    Z = data(:,5)';
    NumberOfInputs = 4;

% ------------------------------------------------
%NORMALIZE
Zmin = min(Z); Zmax = max(Z);

x = (x-[ones(size(x,1),1)*Zmin])./([ones(size(x,1),1)*Zmax]-[ones(size(x,1),1)*Zmin]);
y = (y-[ones(size(y,1),1)*Zmin])./([ones(size(y,1),1)*Zmax]-[ones(size(y,1),1)*Zmin]);
% X = []; y =[]; k=0; loop = 1;
 X = x; 
% while loop
%     k = k + 1;
%     X = [X; x(k+0:k+NumberOfInputs-1)];
%     y = [y; x(k+NumberOfInputs)];
%     if k+NumberOfInputs >= LengthOfTimeSeries; loop = 0; end
% end

num_data = length(y);

train_index = 1:2:num_data;
val_index = 2:2:num_data;
train_in = X(train_index,:);
train_out = y(train_index,:);
val_in = X(val_index,:);
val_out = y(val_index,:);
num_trdata = size(train_in,1);
num_valdata = size(val_in,1);





N = 2; % Subset generation size
MAX_NEURONS = num_trdata - 1 ; % Max RBFS

err_tr = [];
err_val = [];
stop_condition = 1e-7;

fValBest = inf;
fValid = [];
for K = 1:MAX_NEURONS
    K
    % ----- generating rbf set -----
    [G1 centers sigmas] = generate_rbfs(train_in, N);
%     input("s")
%     figure(1)
%     plot3(centers(1,:),centers(2,:),centers(3,:),'r*'); hold on;
%     hold off;

    % ----- Training of NN -----
    [selected_rbfs1, W1, E_k1, A_k1, Q_k1, B_k1, centers, sigmas, G1, G2] =  trainParameters(train_in, val_in,train_out, G1, centers, sigmas, K);
    yhat_tr = 0; yhat_val=0;
    for i = 1:K
        yhat_tr = yhat_tr + W1(i) * G1(:,selected_rbfs1(i));
    end
    for i = 1:K
        yhat_val = yhat_val + W1(i) * G2(:,selected_rbfs1(i));
    end
    err_tr(K,1) = sum((train_out - yhat_tr).^2) / length(train_out);
    err_val(K,1) = sum((val_out - yhat_val).^2) / length(val_out);
    figure(2)
    plot(1:1:K,err_tr(1:K,1),'r')
    hold on
    plot(1:1:K,err_val(1:K,1),'--b')
    fValid = [fValid;log10(err_val(K,1))];
    if err_val(K,1)< fValBest
            fValBest = err_val(K,1);
            W_Best = W1;
            centers_Best = centers;
            sigmas_Best = sigmas;
            selected_rbfs_Best = selected_rbfs1;
            K_Best = K;
    else 
            break;
     end
    if (err_tr(K,1) < stop_condition)
        break;
    end
end
hold off

% yhat_tr = 0;
% yhat_tr = zeros(1,LengthOfTimeSeries);
% for j = 1:1000
%     for i = 1:K
%     yhat_tr(j) = yhat_tr(j) + W1(i) * RBFIO(x(j,:), sigmas(selected_rbfs1(i)), centers(:,selected_rbfs1(i))');
%     end
%     yhat_tr(j) = yhat_tr(j);
% end
%   yhat_tr = Zmin + (Zmax - Zmin)*yhat_tr;
%   plot(500:600,yhat_tr(500:600),'b'); hold on
%   plot(500:600,Z(500:600),'--r')

subplot(2,2,1);
plot(1:1:K,err_tr(1:K,1),'r');hold on
plot(1:1:K,err_val(1:K,1),'--b');
xline(K_Best,'--g');
legend('Training Error','Validation Error','Best Validation');
xlabel('Number of Neurons') ;
ylabel('MSE') ;
hold off
subplot(2,2,2);
bar(E_k1');

subplot(2,2,[3,4]);
testIndex = LengthOfTimeSeries+1:LengthOfTimeSeries+PredictHorizon;
plot(testIndex,Z(testIndex),'-ro');hold on
%  input = [Z(LengthOfTimeSeries-NumberOfInputs+1:LengthOfTimeSeries)];
Predictions = [];
Input = x_test;
Input = [Input-Zmin]/[Zmax - Zmin];
% Predictions = [input(1)];
yhat_tr = 0;
yhat_tr = zeros(1,PredictHorizon);
% yhat_tr(1) = input(1);
% for j = 1:PredictHorizon
%     for i = 1:K
%     yhat_tr(j) = yhat_tr(j) + W1(i) * RBFIO(input, sigmas(selected_rbfs1(i)), centers(:,selected_rbfs1(i))');
%     end
%     Predictions = [Predictions, yhat_tr(j)];
%     input = [input(2:end), yhat_tr(j)];Z
% end


for j = 1:PredictHorizon
    for i = 1:K_Best
    yhat_tr(j) = yhat_tr(j) + W_Best(i) * RBFIO(Input(j,:), sigmas_Best(selected_rbfs_Best(i)), centers_Best(:,selected_rbfs_Best(i))');
    end
    Predictions = [Predictions, yhat_tr(j)];
end
%DENORMALIZE
Predictions = Zmin + (Zmax - Zmin)*Predictions;
title(['Predictions for ',num2str(PredictHorizon),' Test Data']);
plot(testIndex,Predictions,'-bx');
legend('Desired','Predicted');
