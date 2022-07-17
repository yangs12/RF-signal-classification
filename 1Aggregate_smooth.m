close all; clear; clc

load_filename = 'E:\Data\';
save_filename = fileparts(pwd);
save_filename = [save_filename '\Data\'];

%% Parameters
BUI{1,1} = {'00000'};                         % BUI of RF background activities
BUI{1,2} = {'10000','10001','10010','10011'}; % BUI of the Bebop drone RF activities
BUI{1,3} = {'10100','10101','10110','10111'}; % BUI of the AR drone RF activities
BUI{1,4} = {'11000'};                         % BUI of the Phantom drone RF activities
M = 2048; % Total number of frequency bins
L = 1e5;  % Total number samples in a segment
Q = 10;   % Number of returning points for spectral continuity

%% Main
for opt = 1:length(BUI)   % 4 rows of BUI
    % Load data
    for b = 1:length(BUI{1,opt})
        disp(BUI{1,opt}{b})
        % decide number of segments
        if(strcmp(BUI{1,opt}{b},'00000'))  
            N = 40; % Number of segments for RF background activities
        elseif(strcmp(BUI{1,opt}{b},'10111'))
            N = 17;
        else
            N = 20; % Number of segments for drones RF activities
        end
        data = [];
        data1=[];
        data2=[];
        cnt = 1;
        for n = 0:N
            % Load low-frequency data
            x = csvread([load_filename BUI{1,opt}{b} 'L_' num2str(n) '.csv']);
            % Load high-frequency data
            y = csvread([load_filename BUI{1,opt}{b} 'H_' num2str(n) '.csv']);
            % Re-segment to gain more samples
            for i = 1:length(x)/L
                st = 1 + (i-1)*L;   
                fi = i*L;  
                xf = abs(fftshift(fft(x(st:fi)-mean(x(st:fi)),M))); xf = xf(end/2+1:end); 
                % Moving average filter
                xf=smooth(xf,20);
                
                yf = abs(fftshift(fft(y(st:fi)-mean(y(st:fi)),M))); yf = yf(end/2+1:end);
                yf=smooth(yf,40);
                
                data1(:,cnt) = [xf];
                data2(:,cnt)=[yf];
                cnt = cnt + 1;
            end
            disp(100*n/N)
        end   
        data1 = data1./max(max(data1)); % Low frequency
        data2 = data2./max(max(data2)); % High frequency
        data=[data1; data2];
        Data = data.^2;
        save([save_filename BUI{1,opt}{b} '.mat'],'Data');
    end
end