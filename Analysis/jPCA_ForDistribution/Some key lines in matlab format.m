% loading libraries
addpath 'fromMarksLibraries' -END
addpath 'CircStat2010d' -END

% loading data
load exampleData

% Notes on format of data:
% You must organize your data the way that the example structure 'Data' is organized
% Data should be a structure that is at least one element long.  For the example, 'Data' is 108
% elements long, one for each condition (reach type) that the monkey performed.  If you have only
% one condition (e.g., when we analyze a 30 second period of walking) then you will just have one
% element.
% Data.A should be a matrix, with time running vertically and neurons running horizontally.
%       This is the same format as when using matlabs 'princomp'.
% Data.times should be some set of times that you understand.  You will ask for the analysis / plots to
% apply to subsets of these times.


% these will be used for everything below
jPCA_params.softenNorm = 5;  % how each neuron's rate is normized, see below
jPCA_params.suppressBWrosettes = true;  % these are useful sanity plots, but lets ignore them for now
jPCA_params.suppressHistograms = true;  % these are useful sanity plots, but lets ignore them for now


%% EX1: FIRST PLANE
% plotting the first jPCA plane for 200 ms of data, using 6 PCs (the default)
times = -50:10:150;  % 50 ms before 'neural movement onset' until 150 ms after
jPCA_params.numPCs = 6;  % default anyway, but best to be specific
[Projection, Summary] = jPCA_2(Data, times, jPCA_params);

phaseSpace(Projection, Summary);  % makes the plot

printFigs(gcf, '.', '-dpdf', 'Basic jPCA plot');  % prints in the current directory as a PDF


%% EX2: GREATER RANGE OF TIME
times = -50:10:300;  % 50 ms before 'neural movement onset' until 300 ms after
jPCA_params.numPCs = 6;  % sticking with 6 for now, but will move up to 10 below
[Projection, Summary] = jPCA_2(Data, times, jPCA_params);
phaseSpace(Projection, Summary);  % makes the plot


%% EX3:  also do in higher D
times = -50:10:300;
jPCA_params.numPCs = 12;  % search for the jPCA plane within the top 12 PCs, not just the top 6
[Projection, Summary] = jPCA_2(Data, times, jPCA_params);
phaseSpace(Projection, Summary);  % makes the plot


%% EX4: THREE PLANES

times = -50:10:300;  % first we will just run the analysis as we did above
jPCA_params.numPCs = 12;
[Projection, Summary] = jPCA(Data, times, jPCA_params);
 
% now we will plot all three planes
plotParams.planes2plot = [1 2 3];
phaseSpace(Projection, Summary, plotParams);  % makes all three plots



%% EX5: SIMPLE MOVIE FOR VIEWING

times = -50:10:300;  % This is just what we did for examples 3 & 4 above
jPCA_params.numPCs = 12;
[Projection, Summary] = jPCA(Data, times, jPCA_params);

phaseMovie(Projection, Summary);

% for a greater range of times
movParams.times = -50:10:550;  % a range of times that is broader than that used to find the jPCA projection
phaseMovie(Projection, Summary, movParams);


%% EX6: SAVING MOVIES

% Probably best with an undocked figure.
% You will have to play with '.pixelsToGet'
% You may get a warning if you use to many pixels.

movParams.times = -50:10:550;
movParams.pixelsToGet = [25 35 280 280]; % These WILL vary depending on your monitor and layout.
MV = phaseMovie(Projection, Summary, movParams);

figure; movie(MV);  % shows the movie in a matlab figure window

movie2avi(MV, 'jPCA movie', 'FPS', 12, 'compression', 'none'); % 'MV' now contains the movie




%% EX7: MOVIE OF THE CHANGE IN BASIS FROM PCA TO jPCA

% going back to the original 6D analysis over a short period of time.
times = -50:10:150;  % 50 ms before 'neural movement onset' until 150 ms after
jPCA_params.numPCs = 6;  % default anyway, but best to be specific
[Projection, Summary] = jPCA(Data, times, jPCA_params);

phaseSpace(Projection, Summary);  % Static plot as a sanity check

rotationMovie(Projection, Summary, -50:10:150, 70, 70);  % 70 steps to full rotation.  Show all 70.




% List of examples:

% 3) Plotting a movie.
% 5) Plotting with no mean subtraction.

% errors to avoid:

% After running the examples below, type 'help functionName' for all the relevant functions: jPCA,
% phaseSpace & phaseMovie.  All these functions allow many useful parameters to be passed if you so wish. 





%% EX8: MEAN SUBTRACTION

% turning off mean subtraction
times = -50:10:150;  % 200 ms of time, as in supp fig. 7
jPCA_params.meanSubtract = false;  % no mean subtraction
jPCA_params.numPCs = 10;  % as in supp fig. 7
[Projection, Summary] = jPCA(Data, times, jPCA_params);

plotParams.plotPlanEllipse = false;
plotParams.planes2plot = [1 2 3];
phaseSpace(Projection, Summary, plotParams);  


% turning mean subtraction back on
times = -50:10:150;
jPCA_params.meanSubtract = true;  % the default
jPCA_params.numPCs = 6;  % as in fig 3f
[Projection, Summary] = jPCA(Data, times, jPCA_params);
plotParams.crossCondMean = true;
plotParams.planes2plot = 1;
phaseSpace(Projection, Summary, plotParams);


