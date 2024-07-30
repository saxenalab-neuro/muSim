
% not sure if this will work
function rotate2jPCA(Projection, Summary, times, totalSteps, step, reusePlot)

numConds = length(Projection);

for c = 1:numConds
    data = Projection(c).tradPCAprojAllTimes;
    
    dataRot = data * (Summary.jPCs)^(step/totalSteps);
    
    Projection(c).projAllTimes = real(dataRot);  % hijack this field so that we can easily plot it
end


params.reusePlot = reusePlot;
params.times = times;
params.plotPlanEllipse = false;
params.useLabel = false;
params.useAxes = false;
params.planMarkerSize = 7.5;
phaseSpace(Projection, Summary, params);