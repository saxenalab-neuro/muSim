clc
clear all
close all

%Make sure that usim test_data folder is included in the MATLAB path
%test_data folder is where the jpca is saved during usim test

%Parameters
%Sampling rate in ms
sampling_rate = 10;   %Sampling rate in ms; we do jpca at 10ms sampling rate as done previously in original jpca implementation
numPCs = 4;   % Number of PCs for jpca
end_tpoint= 620;   %timepoints to do the jpca
%Select the cycle for each condition to do the jpca
cycle = [3, 3, 3, 3, 3, 3];

load('Data_jpca.mat')
load('n_fixedsteps_jpca.mat')
load('condition_tpoints_jpca.mat')
%%
%iterate through the Data object and format
for i= 1:size(Data, 2)
    
    Data(i).A = double(Data(i).A.A);
    Data(i).times = double(Data(i).times.times');

end

%%


min_tpoints_cond = inf;
for i= 1:size(Data, 2)
    
    Data(i).A = Data(i).A(n_fsteps(i) + cycle(i) * cond_tpoints(i): n_fsteps(i) + (cycle(i)+1) * cond_tpoints(i), :);
    
    Data(i).times = Data(i).times(n_fsteps(i) + cycle(i) * cond_tpoints(i): n_fsteps(i) + (cycle(i)+1) * cond_tpoints(i), :);
    Data(i).times = Data(i).times - Data(i).times(1);
    
    %Now sample the data at the required sampling rate
    Data(i).A = Data(i).A(1:sampling_rate:length(Data(i).A), :);
    Data(i).times = Data(i).times(1:sampling_rate:length(Data(i).times), :);
    
    if min_tpoints_cond > size(Data(i).A, 1)
        
        min_tpoints_cond = size(Data(i).A, 1);
    
    end
    
end

%Now select the min_tpoints for all the conditions for jpca analysis
for i=1:size(Data, 2)
    
    Data(i).A = Data(i).A(1:min_tpoints_cond, :);
    Data(i).times = Data(i).times(1:min_tpoints_cond, :);

end
%%

%Now the data array has been created. Apply the jPCA to obtain the
%corresponding plot


%%-------------------------------------------------------------------------

% these will be used for everything below
jPCA_params.softenNorm = 1;  % how each neuron's rate is normized, see below
jPCA_params.suppressBWrosettes = true;  % these are useful sanity plots, but lets ignore them for now
jPCA_params.suppressHistograms = false;  % these are useful sanity plots, but lets ignore them for now
jPCA_params.meanSubtract = false;

%% EX1: FIRST PLANE
times = 0:sampling_rate:end_tpoint;  % 0 ms before 'movement onset' until 150 ms after
jPCA_params.numPCs = 4;  % default anyway, but best to be specific
[Projection, Summary] = jPCA_4(Data, times, jPCA_params);

phaseSpace(Projection, Summary);  % makes the plot

printFigs(gcf, './jpca_plo;t', '-dpdf', 'jPCA_Plot');  % prints in the current directory as a PDF