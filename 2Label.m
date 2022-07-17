close all; clear; clc
load_filename = 'E:\Data\'; 
save_filename = load_filename;

%% Parameters
BUI{1,1} = {'00000'};                         % BUI of RF background activities
BUI{1,2} = {'10000','10001','10010','10011'}; % BUI of the Bebop drone RF activities
BUI{1,3} = {'10100','10101','10110','10111'}; % BUI of the AR drone RF activities
BUI{1,4} = {'11000'};                         % BUI of the Phantom drone RF activities

%% Loading and concatenating RF data
T = length(BUI);  % four rows of BUI
DATA = [];
LN   = [];
for t = 1:T
    for b = 1:length(BUI{1,t})   
        load([load_filename BUI{1,t}{b} '.mat'])
        DATA = [DATA, Data];
        LN   = [LN size(Data,2)];  % number of column in Data. [Column col...]
        clear Data;
    end
    disp(100*t/T)
end

%% Label dependent on category number
zeros(3,sum(LN));
Label(1,:) = [0*ones(1,LN(1)) 1*ones(1,sum(LN(2:end)))];   % only 00000 is no drone, 1 for other
Label(2,:) = [0*ones(1,LN(1)) 1*ones(1,sum(LN(2:5))) 2*ones(1,sum(LN(6:9))) 3*ones(1,LN(10))];  % every kind of drone
temp = [];
for i = 1:length(LN)
    temp = [temp (i-1)*ones(1,LN(i))];  % label 0-9 kinds
end
Label(3,:) = temp;

%% Saving
csvwrite([save_filename 'RF_Datasmooth.csv'],[DATA; Label]);
