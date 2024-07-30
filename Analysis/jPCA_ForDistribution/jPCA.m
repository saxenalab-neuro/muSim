
% [Projection, Summary] = jPCA(Data, analyzeTimes, params)
%
% OUTPUTS:
%   Projection is a struct with one element per condition.
%   It contains the following fields:
%       .proj           The projection into the 6D jPCA space.
%       .times          Those times that were used
%       .projAllTimes   Pojection of all the data into the jPCs (which were derived from a subset of the data)
%       .allTimes       All the times (exactly the same as Data(1).times)
%       .tradPCAproj    The traditional PCA projection (same times as for 'proj')
%       .tradPCAprojAllTimes   Above but for all times.
%
%   Summary contains the following fields:
%       .jPCs           The jPCs in terms of the PCs (not the full-D space)
%       .PCs            The PCs (each column is a PC, each row a neuron)
%       .jPCs_highD     The jPCs, but in the original high-D space.  This is just PCs * jPCs, and is thus of size neurons x numPCs
%       .varCaptEachJPC The data variance captured by each jPC
%       .varCaptEachPC  The data variance captured by each PC
%       .R2_Mskew_2D    Fit quality (fitting dx with x) provided by Mskew, in the first 2 jPCs.
%       .R2_Mbest_2D    Same for the best M
%       .R2_Mskew_kD    Fit quality (fitting dx with x) provided by Mskew, in all the jPCs (the number of which is set by 'numPCs'.
%       .R2_Mbest_kD    Same for the best M
%
%       There are some other useful currently undocumented (but often self-explanatory) fields in
%       'Summary'
%
%   Summary also contains the following, useful for projecting new data into the jPCA space.  Also
%   useful for getting from what is in 'Projection' back into the high-D land of PSTHs
%   Note that during preprocessing we FIRST normalize, and then (when doing PCA) subtract the
%   overall mean firing rate (no relationship to the cross-condition mean subtraction).  For new
%   data you must do this in the same order (using the ORIGINAL normFactors and mean firing rats.
%   To get from low-D to high D you must do just the reverse: add back the mean FRs and the multiply
%   by the normFactors.
%       .preprocessing.normFactors = normFactors;  
%       .preprocessing.meanFReachNeuron = meanFReachNeuron; 
%
% INPUTS:
% The input 'Data' needs to be a struct, with one entry per condition.
% For a given condition, Data(c).A should hold the data (e.g. firing rates).
% Each column of A corresponds to a neuron, and each row to a timepoint.
%
% Data(c).times is an optional field.  If you provide it, only those entries that match
% 'analyzeTimes' will be used for the analysis. If  analyzeTimes == [], all times will be used.
%  
%  If you don't provide it, a '.times' field
% is created that starts at 1.  'analyzeTimes' then refers to those times.
% 
% 'params' is optional, and can contain the following fields:
%   .numPCs        Default is 6. The number of traditional PCs to use (all jPCs live within this space)
%   .normalize     Default is 'true'.  Whether or not to normalize each neurons response by its FR range.
%   .softenNorm    Default is 10.  Determines how much we undernormalize for low FR neurons.  0 means
%                  complete normalization.  10 means a neuron with FR range of 10 gets mapped to a range of 0.5.
%   .meanSubtract  Default is true.  Whether or not we remove the across-condition mean from each
%                  neurons rate.
%   .suppressBWrosettes    if present and true, the black & white rosettes are not plotted
%   .suppressHistograms    if present and true, the blue histograms are not plotted
%   .suppressText          if present and true, no text is output to the command window 
%
%  As a note on projecting more data into the same space.  This can be done with the function
%  'projectNewData'.  However, if you are going to go this route then you should turn OFF
%  'meanSubtract'.  If you wish to still mean subtract the data you should do it by hand yourself.
%  That way you can make a principled decision regarding how to treat the original data versus the
%  new-to-be-projected data.  For example, you may subtract off the mean manually across 108
%  conditions.  You might then leave out one condition and compute the jCPA plane (with meanSubtract set to false).  
%  You could then project the remaining condition into that plane using 'projectNewData'.
% 
% 
function [Projection, Summary] = jPCA(Data, analyzeTimes, params)


%% quick bookkeeping

numConds = length(Data);
numTimes = size(Data(1).A,1);
% there is also a 'numAnalyzedTimes' defined below.

%% setting parameters that may or may not have been specified

% 'times' field
% if the user didn't specify a times field, create one that starts with '1'
if ~isfield(Data(1),'times')
    for c = 1:length(Data)
        Data(c).times = 1:numTimes;
    end
end

if exist('analyzeTimes', 'var') && ~isempty(analyzeTimes) && max(diff(analyzeTimes)) > max(diff(Data(1).times))
    disp('error, you can use a subset of times but you may not skip times within that subset');
    Projection = []; Summary = []; return;
end

% the number of PCs to look within
numPCs = 6;
if exist('params', 'var') && isfield(params,'numPCs')
    numPCs = params.numPCs;
end
if rem(numPCs,2)>0
    disp('you MUST ask for an even number of PCs.'); return;
end

% do we normalize
normalize = true;
if exist('params', 'var') && isfield(params,'normalize')
    normalize = params.normalize;
end

% do we soften the normalization (so weak signals stay smallish)
% numbers larger than zero mean soften the norm.
% The default (10) means that 10 spikes a second gets mapped to 0.5, infinity to 1, and zero to zero.
% Beware if you are using data that isn't in terms of spikes/s, as 10 may be a terrible default
softenNorm = 10;
if exist('params', 'var') && isfield(params,'softenNorm')
    softenNorm = params.softenNorm;
end

% do we mean subtract
meanSubtract = true;
if exist('params', 'var') && isfield(params,'meanSubtract')
    meanSubtract = params.meanSubtract;
end
if length(Data)==1, meanSubtract = false; end  % cant mean subtract if there is only one condition

if ~exist('analyzeTimes', 'var') || isempty(analyzeTimes)
    disp('analyzing all times');
    analyzeTimes = Data(1).times;
end

%% figure out which times to analyze and make masks
%
analyzeIndices = ismember(round(Data(1).times), analyzeTimes);
if size(analyzeIndices,1) == 1
    analyzeIndices = analyzeIndices';  % orientation matters for the repmat below
end
analyzeMask = repmat(analyzeIndices,numConds,1);  % used to mask bigA
if diff( Data(1).times(analyzeIndices) ) <= 5
    disp('mild warning!!!!: you are using a short time base which might make the computation of the derivative a bit less reliable');
end

% these are used to take the derivative
bunchOtruth = true(sum(analyzeIndices)-1,1);
maskT1 = repmat( [bunchOtruth;false],numConds,1);  % skip the last time for each condition
maskT2 = repmat( [false;bunchOtruth],numConds,1);  % skip the first time for each condition

if sum(analyzeIndices) < 5
    disp('warning, analyzing few or no times');
    disp('if this wasnt your intent, check to be sure that you are asking for times that really exist');
end

%% make a version of A that has all the data from all the conditions.
% in doing so, mean subtract and normalize

bigA = vertcat(Data.A);  % append conditions vertically

% note that normalization is done based on ALL the supplied data, not just what will be analyzed
if normalize  % normalize (incompletely unless asked otherwise)
    ranges = range(bigA);  % For each neuron, the firing rate range across all conditions and times.
    normFactors = (ranges+softenNorm);
    bigA = bsxfun(@times, bigA, 1./normFactors);  % normalize
else
    normFactors = ones(1,size(bigA,2));
end
 


sumA = 0;
for c = 1:numConds
    sumA = sumA + bsxfun(@times, Data(c).A, 1./normFactors);  % using the same normalization as above
end
meanA = sumA/numConds;
if meanSubtract  % subtract off the across-condition mean from each neurons response
    bigA = bigA-repmat(meanA,numConds,1);
end



%% now do traditional PCA

smallA = bigA(analyzeMask,:);
[PCvectors,rawScores] = princomp(smallA,'econ');  % apply PCA to the analyzed times
meanFReachNeuron = mean(smallA);  % this will be kept for use by future attempts to project onto the PCs

% these are the directions in the high-D space (the PCs themselves)
if numPCs > size(PCvectors,2)
    disp('You asked for more PCs than there are dimensions of data');
    disp('Giving up');
end
PCvectors = PCvectors(:,1:numPCs);  % cut down to the right number of PCs

% CRITICAL STEP
% This is what we are really after: the projection of the data onto the PCs
Ared = rawScores(:,1:numPCs);  % cut down to the right number of PCs

% Some extra steps
%
% projection of all the data
% princomp subtracts off means automatically so we need to do that too when projecting all the data
bigAred = bsxfun(@minus, bigA, mean(smallA)) * PCvectors(:,1:numPCs); % projection of all the data (not just teh analyzed chunk) into the low-D space.
% need to subtract off the mean for smallA, as this was done automatically when computing the PC
% scores from smallA, and we want that projection to match this one.

% projection of the mean

meanAred = bsxfun(@minus, meanA, mean(smallA)) * PCvectors(:,1:numPCs);  % projection of the across-cond mean (which we subtracted out) into the low-D space.

% will need this later for some indexing
numAnalyzedTimes = size(Ared,1)/numConds;

%% GET M & Mskew
% compute dState, and use that to find the best M and Mskew that predict dState from the state

% we are interested in the eqn that explains the derivative as a function of the state: dState/dt = M*State
dState = Ared(maskT2,:) - Ared(maskT1,:);  % the masks just give us earlier and later times within each condition
preState = Ared(maskT1,:);  % just for convenience, keep the earlier time in its own variable

% first compute the best M (of any type)
% note, we have converted dState and Ared to have time running horizontally
M = (dState'/preState');  % M takes the state and provides a fit to dState
% Note on sizes of matrices:
% dState' and preState' have time running horizontally and state dimension running vertically 
% We are thus solving for dx = Mx.
% M is a matrix that takes a column state vector and gives the derivative

% now compute Mskew using John's method
% Mskew expects time to run vertically, transpose result so Mskew in the same format as M
% (that is, Mskew will transform a column state vector into dx)
Mskew = skewSymRegress(dState,preState)';  % this is the best Mskew for the same equation

%% USE Mskew to get the jPCs

% get the eigenvalues and eigenvectors
[V,D] = eig(Mskew); % V are the eigenvectors, D contains the eigenvalues
evals = diag(D); % eigenvalues

% the eigenvalues are usually in order, but not always.  We want the biggest
[~,sortIndices] = sort(abs(evals),1,'descend');
evals = evals(sortIndices);  % reorder the eigenvalues
evals = imag(evals);  % get rid of any tiny real part
V = V(:,sortIndices);  % reorder the eigenvectors (base on eigenvalue size)

% Eigenvalues will be displayed to confirm that everything is working
% unless we are asked not to output text
if ~exist('params', 'var') || ~isfield(params,'suppressText') || ~params.suppressText
    disp('eigenvalues of Mskew: ');
    for i = 1:length(evals)
        if evals(i) > 0;
            fprintf('                  %1.3fi', evals(i));
        else
            fprintf('     %1.3fi \n', evals(i));
        end
    end
end

jPCs = zeros(size(V));
for pair = 1:numPCs/2
    vi1 = 1+2*(pair-1);
    vi2 = 2*pair;
    
    VconjPair = V(:,[vi1,vi2]);  % a conjugate pair of eigenvectors
    evConjPair = evals([vi1,vi2]); % and their eigenvalues
    VconjPair = getRealVs(VconjPair,evConjPair);
    
    jPCs(:,[vi1,vi2]) = VconjPair;
end

%% Get the projections

proj = Ared * jPCs;
projAllTimes = bigAred * jPCs;
tradPCA_AllTimes = bsxfun(@minus, bigA, mean(smallA)) * PCvectors;  % mean center in exactly the same way as for the shorter time period.
crossCondMeanAllTimes = meanAred * jPCs;

% Do some annoying output formatting.
% Put things back so we have one entry per condition
index1 = 1;
index2 = 1;
for c = 1:numConds
    index1b = index1 + numAnalyzedTimes -1;  % we will go from index1 to this point
    index2b = index2 + numTimes -1;  % we will go from index2 to this point
    
    Projection(c).proj = proj(index1:index1b,:);
    Projection(c).times = Data(1).times(analyzeIndices);
    Projection(c).projAllTimes = projAllTimes(index2:index2b,:);
    Projection(c).allTimes = Data(1).times;
    Projection(c).tradPCAproj = Ared(index1:index1b,:);
    Projection(c).tradPCAprojAllTimes = tradPCA_AllTimes(index2:index2b,:);
    
    index1 = index1+numAnalyzedTimes;
    index2 = index2+numTimes;
end
   
%% Done computing the projections, plot the rosette

% do this unless params contains a field 'suppressBWrosettes' that is true
if ~exist('params', 'var') || ~isfield(params,'suppressBWrosettes') || ~params.suppressBWrosettes
    plotRosette(Projection, 1);  % primary plane
    plotRosette(Projection, 2);  % secondary plane
end


%% SUMMARY STATS
%% compute R2 for the fit provided by M and Mskew

% R2 Full-D
fitErrorM = dState'- M*preState';
fitErrorMskew = dState'- Mskew*preState';
varDState = sum(dState(:).^2);  % original data variance

R2_Mbest_kD = (varDState - sum(fitErrorM(:).^2)) / varDState;  % how much is explained by the overall fit via M
R2_Mskew_kD = (varDState - sum(fitErrorMskew(:).^2)) / varDState;  % how much by is explained via Mskew

% unless asked to not output text
if ~exist('params', 'var') || ~isfield(params,'suppressText') || ~params.suppressText
    fprintf('%% R^2 for Mbest (all %d dims):   %1.2f\n', numPCs, R2_Mbest_kD);
    fprintf('%% R^2 for Mskew (all %d dims):   %1.2f  <<---------------\n', numPCs, R2_Mskew_kD);
end


% R2 2-D primary jPCA plane
fitErrorM_2D = jPCs(:,1:2)' * fitErrorM;  % error projected into the primary plane
fitErrorMskew_2D = jPCs(:,1:2)' * fitErrorMskew;  % error projected into the primary plane
dState_2D = jPCs(:,1:2)' * dState'; % project dState into the primary plane
varDState_2D = sum(dState_2D(:).^2); % and get its variance

R2_Mbest_2D = (varDState_2D - sum(fitErrorM_2D(:).^2)) / varDState_2D;  % how much is explained by the overall fit via M
R2_Mskew_2D = (varDState_2D - sum(fitErrorMskew_2D(:).^2)) / varDState_2D;  % how much by is explained via Mskew

if ~exist('params', 'var') || ~isfield(params,'suppressText') || ~params.suppressText
    fprintf('%% R^2 for Mbest (primary 2D plane):   %1.2f\n', R2_Mbest_2D);
    fprintf('%% R^2 for Mskew (primary 2D plane):   %1.2f  <<---------------\n', R2_Mskew_2D);
end

%% variance catpured by the jPCs
origVar = sum(sum( bsxfun(@minus, smallA, mean(smallA)).^2));
varCaptEachPC = sum(Ared.^2) / origVar;  % this equals latent(1:numPCs) / sum(latent)
varCaptEachJPC = sum((Ared*jPCs).^2) / origVar;
varCaptEachPlane = reshape(varCaptEachJPC, 2, numPCs/2);
varCaptEachPlane = sum(varCaptEachPlane);


%% Analysis of whether things really look like rotations (makes plots)

for jPCplane = 1:2
    phaseData = getPhase(Projection, jPCplane);  % does the key analysis
    
    if exist('params', 'var')
        cstats = plotPhaseDiff(phaseData, params, jPCplane);  % plots the histogram.  'params' is just what the user passed, so plots can be suppressed
    else
        cstats = plotPhaseDiff(phaseData, [], jPCplane);
    end
    
    if jPCplane == 1
        circStats = cstats;  % keep only for the primary plane
    end
end

%% Make the summary output structure

Summary.jPCs = jPCs;
Summary.PCs = PCvectors;
Summary.jPCs_highD = PCvectors * jPCs;
Summary.varCaptEachJPC = varCaptEachJPC;
Summary.varCaptEachPC = varCaptEachPC;
Summary.varCaptEachPlane = varCaptEachPlane;
Summary.Mbest = M;
Summary.Mskew = Mskew;
Summary.fitErrorM = fitErrorM;
Summary.fitErrorMskew = fitErrorMskew;
Summary.R2_Mskew_2D = R2_Mskew_2D;
Summary.R2_Mbest_2D = R2_Mbest_2D;
Summary.R2_Mskew_kD = R2_Mskew_kD;
Summary.R2_Mbest_kD = R2_Mbest_kD;
Summary.circStats = circStats;
Summary.acrossCondMeanRemoved = meanSubtract;
Summary.crossCondMean = crossCondMeanAllTimes(analyzeIndices,:);
Summary.crossCondMeanAllTimes = crossCondMeanAllTimes;
Summary.preprocessing.normFactors = normFactors;  % Used for projecting new data from the same neurons into the jPC space
Summary.preprocessing.meanFReachNeuron = meanFReachNeuron; % You should first normalize and then mean subtract using this (the original) mean
% conversely, to come back out, you must add the mean back on and then MULTIPLY by the normFactors
% to undo the normalization.

%% DONE (only functions below)


%% Inline function that gets the real analogue of the eigenvectors
function Vr = getRealVs(V,evals)

    % get real vectors made from the eigenvectors
    
    % by paying attention to this order, things will always rotate CCW
    if abs(evals(1))>0  % if the eigenvalue with negative imaginary component comes first
        Vr = [V(:,1) + V(:,2), (V(:,1) - V(:,2))*1i]; 
    else
        Vr = [V(:,2) + V(:,1), (V(:,2) - V(:,1))*1i];
    end
    Vr = Vr / sqrt(2);

    % now get axes aligned so that plan is spread mostly along the horizontal axis
    testProj = (Vr'*Ared(1:numAnalyzedTimes:end,:)')'; % just picks out the plan times
    rotV = princomp(testProj);
    crossProd = cross([rotV(:,1);0], [rotV(:,2);0]);
    if crossProd(3) < 0, rotV(:,2) = -rotV(:,2); end   % make sure the second vector is 90 degrees clockwise from the first
    Vr = Vr*rotV; 

    % flip both axes if necessary so that the maximum move excursion is in the positive direction
    testProj = (Vr'*Ared')';  % all the times
    if max(abs(testProj(:,2))) > max(testProj(:,2))  % 2nd column is the putative 'muscle potent' direction.
        Vr = -Vr;
    end
end

end
%% end of main function


%% Some inline functions are below

%% Plot the rosette itself (may need to spruce this up and move to a subfunction)
function plotRosette(Proj, whichPair)

    d1 = 1 + 2*(whichPair-1);
    d2 = d1+1;

    numConds = length(Proj);

    figure;

    % first deal with the ellipse for the plan variance (we want this under the rest of the data)
    planData = zeros(numConds,2);
    for c = 1:numConds
        planData(c,:) = Proj(c).proj(1,[d1,d2]);
    end
    planVars = var(planData);
    circle([0 0], 2*planVars.^0.5, 0.6*[1 1 1], 1); hold on;
    %fprintf('ratio of plan variances = %1.3f (hor var / vert var)\n', planVars(1)/planVars(2));

    allD = vertcat(Proj(:).proj);  % just for getting axes
    allD = allD(:,d1:d2);
    mxVal = max(abs(allD(:)));
    axLim = mxVal*1.05*[-1 1 -1 1];
    arrowSize = 5;
    for c = 1:numConds
        plot(Proj(c).proj(:,d1), Proj(c).proj(:,d2), 'k');
        plot(Proj(c).proj(1,d1), Proj(c).proj(1,d2), 'ko', 'markerFaceColor', [0.7 0.9 0.9]);

        penultimatePoint = [Proj(c).proj(end-1,d1), Proj(c).proj(end-1,d2)];
        lastPoint = [Proj(c).proj(end,d1), Proj(c).proj(end,d2)];
        arrowMMC(penultimatePoint, lastPoint, [], arrowSize, axLim);

    end

    axis(axLim);
    axis square;
    plot(0,0,'k+');
    
    title(sprintf('jPCA plane %d', whichPair));
end


%% Getting the phases
function phaseData = getPhase(Proj, whichPair)
    numConds = length(Proj);
    d1 = 1 + 2*(whichPair-1);
    d2 = d1+1;
    
    for c=1:numConds
        data = Proj(c).proj(:,[d1,d2]);
        phase = atan2(data(:,2), data(:,1));  % Y comes first for atan2
        
        deltaData = diff(data);
        phaseOfDelta = atan2(deltaData(:,2), deltaData(:,1));  % Y comes first for atan2
        phaseOfDelta = [phaseOfDelta(1); phaseOfDelta];  %#ok<AGROW> % so same length as phase
        radius = sum(data.^2,2).^0.5;
        
        % collect and format
        % make things run horizontally so they can be easily concatenated.
        phaseData(c).phase = phase'; %#ok<AGROW>
        phaseData(c).phaseOfDelta = phaseOfDelta'; %#ok<AGROW>
        phaseData(c).radius = radius'; %#ok<AGROW>
        
        % angle between state vector and Dstate vector
        % between -pi and pi
        phaseData(c).phaseDiff = minusPi2Pi(phaseData(c).phaseOfDelta - phaseData(c).phase); %#ok<AGROW>
    end
    
end



%% plotting the phase difference between dx(t)/dt and x(t) where x is the 2D state
function circStatsSummary = plotPhaseDiff(phaseData, params, jPCplane)
    % compute the circular mean of the data, weighted by the r's
    circMn = circ_mean([phaseData.phaseDiff]', [phaseData.radius]');
    resultantVect = circ_r([phaseData.phaseDiff]', [phaseData.radius]');
    
    
    bins = pi*(-1:0.1:1);
    cnts = histc([phaseData.phaseDiff], bins);  % not for plotting, but for passing back out
    

    % do this unless params contains a field 'suppressHistograms' that is true
    if ~exist('params', 'var') || ~isfield(params,'suppressHistograms') || ~params.suppressHistograms
        figure;
        hist([phaseData.phaseDiff], bins); hold on;
        plot(circMn, 20, 'ro', 'markerFa', 'r', 'markerSiz', 8);
        plot(pi/2*[-1 1], [0 0], 'ko', 'markerFa', 'r', 'markerSiz', 8);
        set(gca,'XLim',pi*[-1 1]);
        title(sprintf('jPCs plane %d', jPCplane));
    end
    
    %fprintf('(pi/2 is %1.2f) The circular mean (weighted) is %1.2f\n', pi/2, circMn);
    
    % compute the average dot product of each datum (the angle difference for one time and condition)
    % with pi/2.  Will be one for perfect rotations, and zero for random data or expansions /
    % contractions.
    avgDP = averageDotProduct([phaseData.phaseDiff]', pi/2);
    %fprintf('the average dot product with pi/2 is %1.4f  <<---------------\n', avgDP);
    
    circStatsSummary.circMn = circMn;
    circStatsSummary.resultantVect = resultantVect;
    circStatsSummary.avgDPwithPiOver2 = avgDP;  % note this basically cant be <0 and definitely cant be >1
    circStatsSummary.DISTRIBUTION.bins = bins;
    circStatsSummary.DISTRIBUTION.binCenters = (bins(1:end-1) + bins(2:end))/2;
    circStatsSummary.DISTRIBUTION.counts = cnts(1:end-1);
    circStatsSummary.RAW.rawData = [phaseData.phaseDiff]';
    circStatsSummary.RAW.rawRadii = [phaseData.radius]';
    
end

%% for computing the average dot product with a comparison angle
function avgDP = averageDotProduct(angles, compAngle, varargin)
    
    x = cos(angles-compAngle);
    
    if ~isempty(varargin)
        avgDP = mean(x.*varargin{1}) / mean(varargin{1});  % weighted sum
    else
        avgDP = mean(x);
    end
end
        
















