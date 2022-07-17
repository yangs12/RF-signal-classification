close all; clear; clc
filepath = 'E:\research\DL\UAV\DroneRF-master\DroneRF-master\Python\';

%% Parameters
opt = 3;  % Change to 1, 2, or 3 to alternate between the 1st, 2nd, and 3rd DNN results respectively.

%% Main
y = [];
for i = 1:7
    x = csvread([filepath 'Outputs' num2str(i) '.csv']);
    y = [y ; x];
end

%% Plotting confusion matrix
if(opt == 1)
    plotconfusion_mod(y(:,1:2)',y(:,3:4)');
elseif(opt == 2)
    plotconfusion_mod(y(:,1:4)',y(:,5:8)');
elseif(opt == 3)
    plotconfusion_mod(y(:,1:10)',y(:,11:20)');
    set(gcf,'position',[100, -100, 800, 800])
end
set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));
