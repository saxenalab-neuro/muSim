% For making rosette movies for the paper
% useage:
%
%   phaseMovie(Projection, Summary);    Easiest usage.  Just plot in matlab. Use default params.
%   phaseMovie(Projection, Summary, movieParams);   Can override one or more parameters
%   MV = phaseMovie(Projection, Summary, movieParams);   if you need to save a movie
%
%   To save the movie use: movie2avi(MV, 'movieName', 'FPS', 12, 'compression', 'none');
%
%   'movieParams' can contain the following fields:
%       .plane2plot   Default is 1.  Set to 2 to see the second plane (and so on)
%       .rankType     Default is 'eig': the first plane is that associated with the largest eigenvalue.
%                     Can also be 'varCapt'
%       .times        Default is Projection(1).times.  Note that you can specify a subset of those
%                     times or a superset.  If the latter, only those times that lie within
%                     'allTimes' will be used.
%       .conds2plot   Default is 'all'.  Can also be a scalar (to plot a single cond), or a vector of
%                     conds (e.g., [1 5 12 27];
%       .substRawPCs  Substitute raw PCs.
%       .pixelsToGet  You may wish to customize these, esp. if the defaults don't work well on your
%                     screen.  Co-ordinates are left then bottom, then width then height.
%                     See getframe for more info.  
%       .usePads      Default is 0.  If '1', stationary pads will be added to start and end of
%                     movie.  This can be useful in some media players (though not keynote).
%       .arrowGain    Controls how much the arrow grows with speed.  Default is 25;
%       .tail         If specified and not empty, a tail of this length (in ms) will be produced, instead of the whole trajectory. 


function MV = phaseMovie(Projection, Summary, movieParams)


%% Set defaults and override if fields are set in 'movieParams'


% PLANE
% Plot the first plane (eigenvalue-wise) unless asked to do otherwise
frameParams.planes2plot = 1;
if exist('movieParams', 'var') && isfield(movieParams,'plane2plot')
    frameParams.planes2plot = movieParams.plane2plot(1); % do what we are told, but we can only plot one plane
end
frameParams.rankType = 'eig';  % can also be 'varCapt'
if exist('movieParams', 'var') && isfield(movieParams,'rankType')
    frameParams.rankType = movieParams.rankType;
end


% TIMES
times2plot = Projection(1).times;
if exist('movieParams', 'var') && isfield(movieParams,'times')
    times2plot = movieParams.times;
    times2plot = times2plot(ismember(times2plot, Projection(1).allTimes));  % can only plot what we have access to 
end

% CONDS
frameParams.conds2plot = 'all';
if exist('movieParams', 'var') && isfield(movieParams,'conds2plot')
    frameParams.conds2plot = movieParams.conds2plot;
end

% jPCs or raw PCs
frameParams.substRawPCs = false;
if exist('movieParams', 'var') && isfield(movieParams,'substRawPCs')
    frameParams.substRawPCs = movieParams.substRawPCs;
end

% PIXELS
pixelsToGet = [70 -90 600 590];
if exist('movieParams', 'var') && isfield(movieParams,'pixelsToGet')
    pixelsToGet = movieParams.pixelsToGet;
end

% Stationary Padding
% Default is we do not add any stationary padding to start or end.  But we will if asked
usePads = false;
if exist('movieParams', 'var') && isfield(movieParams,'usePads')
    usePads = movieParams.usePads;
end
% These are used only if the above flag is true
stationaryPadStart = 18;  % extra frames at the beginning of stationary image.
stationaryPadEnd = 24;  % extra frames at the end of stationary image.

% Arrow size changes with speed
frameParams.arrowGain = 25;  % controls how the arrow grows with speed
if exist('movieParams', 'var') && isfield(movieParams,'arrowGain')
    frameParams.arrowGain = movieParams.arrowGain;
end

% Tail
tail = [];
if exist('movieParams', 'var') && isfield(movieParams,'tail')
    tail = movieParams.tail;
end



% These plotting parameters are currently hard coded (can't be changed by movieParams)
% If needed, one can of course 
frameParams.arrowSize = 3.3;  % will likely grow
frameParams.planMarkerSize = 6.5;
frameParams.lineWidth = 0.5;
frameParams.useAxes = 0;
frameParams.useLabel = 0;
frameParams.plotPlanEllipse = 0;
frameParams.reusePlot = 0;  % will change to one after first frame

if ~isempty(tail)
    frameParams.planMarkerSize = 0;  % not plan marker if we are using a non-infinite tail
end
    

%% Done handling parameters and defaults
% Print some things out to confirm to the user the choices being made
fprintf('using plane %d (ordered by %s)\n',  frameParams.planes2plot, frameParams.rankType);
fprintf('movie runs from time %d to %d\n',  times2plot([1,end]));
fprintf('pixels: [%d %d %d %d]\n',  pixelsToGet);


%% Now start plotting stuff

fi = 1;  % frame index

%% start pad
if nargout > 0  && usePads == 1
    % pad at start if we are making the  movie to export, rather than just view in matlab

    for i = 1:stationaryPadStart
        frameParams.times = times2plot(1):times2plot(2);

        phaseSpace(Projection, Summary, frameParams);
        drawnow;
        frameParams.reusePlot = 1;  % after the first frame, always reuse

        if nargout > 0
            MV(fi) = getframe(gca, pixelsToGet);
            fi=fi+1;
        end
    end
end

%% ** ACTUAL MOVIE **
for ti = 2:length(times2plot)
    frameParams.times = times2plot(1):times2plot(ti);  % only times that match one of these will be used
    if ~isempty(tail)  && range(frameParams.times) > tail
        frameParams.times = frameParams.times(frameParams.times > frameParams.times(end)-tail);
    end
        
    phaseSpace(Projection, Summary, frameParams);
    drawnow;
    frameParams.reusePlot = 1;  % after the first frame, always reuse
    
    if nargout > 0
        MV(fi) = getframe(gca, pixelsToGet);
        fi=fi+1;
    end
end

%% end pad
if nargout > 0  && usePads == 1
    % pad at start if we are making the  movie to export, rather than just view in matlab
    for i = 1:stationaryPadEnd
        frameParams.times = times2plot(1):times2plot(end);

        phaseSpace(Projection, Summary, frameParams);
        drawnow;
        frameParams.reusePlot = 1;  % after the first frame, always reuse

        if nargout > 0
            MV(fi) = getframe(gca, pixelsToGet);
            fi=fi+1;
        end
    end
end


if ~exist('MV', 'var')  % if we were not asked to make the movie structure
    MV = [];
end
    


