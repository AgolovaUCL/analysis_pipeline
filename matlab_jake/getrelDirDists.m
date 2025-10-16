% relativeDirection_ctrlDistribution
% ------------------------------------------------------------
% This script computes control distributions of pure head direction 
% (absolute orientation) and relative head direction (to goals/platforms) 
% for each trial type and each platform. Results are saved to a .mat file.
% ------------------------------------------------------------

%% Load data
cd positionalData
load positionalDataByTrialType.mat    % pos: contains smoothed position and head direction per trial type
load platformLocations.mat            % platformLocations: platform identity per timepoint
load frame.mat frameSize              % frameSize: arena dimensions
cd ..
cd physiologyData
cd direction

trialTypes = fieldnames(pos);         % Extract trial type names (e.g. 'train', 'probe')

% Define binning parameters
binSize = 15;                         % spatial bin size in pixels (resolution of grid)
xAxis = 1:binSize:frameSize(1);       % x grid across arena
yAxis = 1:binSize:frameSize(2);       % y grid across arena
angleEdges = 0:15:360;                % angular bins in degrees (15° bins)

%% Loop over trial types
for tt = 1:length(trialTypes)       
    % Concatenate all positions for this trial type
    position = vertcat(pos.(trialTypes{tt})(:).dlc_XYsmooth);
    
    hd = [];     % head direction (all trials concatenated)
    plat = [];   % platform identity (all trials concatenated)
    
    % Loop through all trials in this trial type
    for t = 1:length(platformLocations.(trialTypes{tt}))
        
        % Extract head direction for this trial
        hdTemp = vertcat(pos.(trialTypes{tt})(t).dlc_angle);
        
        % Interpolate missing values (NaNs)
        hdNaN = find(isnan(hdTemp));
        hdIsN = find(~isnan(hdTemp));
        interpPhase = interpPhaseNEW(hdIsN, hdTemp(hdIsN), hdNaN);
        hdTemp(hdNaN) = interpPhase;
        
        % Store concatenated head directions
        hd = [hd; hdTemp];
        
        % Store corresponding platform IDs
        plat = [plat; vertcat(platformLocations.(trialTypes{tt}){t}.body)];
    end
    
    %% Loop over all possible platforms (here assumed 1–61)
    for p = 1:61 
        platInd = find(plat == p);    % find timepoints assigned to platform p
        if isempty(platInd)
            continue                  % skip if no data for this platform
        end
        
        % Extract positions and head directions for this platform
        posPlat = position(platInd,:);
        hdPlat = hd(platInd);
        
        % Remove entries where head direction is NaN
        posPlat(isnan(hdPlat),:) = [];
        hdPlat(isnan(hdPlat)) = [];
        
        % --- Pure head direction distribution (absolute orientation) ---
        purDirDistsTemp = histcounts(hdPlat, angleEdges);
        purDirDistsTemp = purDirDistsTemp ./ sum(purDirDistsTemp); % normalize
        purDirDists.(trialTypes{tt}){p} = purDirDistsTemp;    
        
        % --- Relative head direction distribution (to goal/grid) ---
        relDirDistsTemp = getRelDirDist(posPlat, hdPlat, xAxis, yAxis, angleEdges);
        relDirDists.(trialTypes{tt}){p} = relDirDistsTemp;
    end
end

%% Save results
save mrlFocus_ctrlDistribution_coarse.mat purDirDists relDirDists angleEdges


%% ------------------------------------------------------------------------
function relDirDist = getRelDirDist(pos, hd, xAxis, yAxis, angleEdges)
% GETRELDIRDIST
% Computes the relative direction distribution between actual head 
% direction and direction-to-goal for each spatial bin.
%
% Inputs:
%   pos        - [N x 2] spike positions (x,y)
%   hd         - [N x 1] head directions in degrees
%   xAxis,yAxis- grid bin centers
%   angleEdges - angular bin edges for histogram
%
% Output:
%   relDirDist - [xBins x yBins x angleBins] relative direction distributions

% Compute X distance from each spike to each bin
xDistance = (xAxis - pos(:,1))';
xDistance = reshape(xDistance, [length(xAxis), 1, length(hd)]);
xDistance = repmat(xDistance, 1, length(yAxis), 1);

% Compute Y distance from each spike to each bin
yDistance = (yAxis - pos(:,2))';
yDistance = reshape(yDistance, [1, length(yAxis), length(hd)]);
yDistance = repmat(yDistance, length(xAxis), 1, 1);

% Compute angle from spike to bin center
dir2goal = rad2deg(atan2(xDistance, yDistance));
dir2goal(dir2goal < 0) = dir2goal(dir2goal < 0) + 360;

% Expand head direction array to match bin grid
spikeHD_XP = reshape(hd, 1, 1, length(hd));
spikeHD_XP = repmat(spikeHD_XP, length(xAxis), length(yAxis));

% Relative direction = head direction – direction to bin
dirRel2Goal = spikeHD_XP - dir2goal;
dirRel2Goal(dirRel2Goal < 0) = 360 + dirRel2Goal(dirRel2Goal < 0);

% Compute relative direction histogram for each spatial bin
relDirDist = NaN(length(xAxis), length(yAxis), length(angleEdges)-1);
for x = 1:length(xAxis)
    for y = 1:length(yAxis)
        distTemp = histcounts(dirRel2Goal(x,y,:), angleEdges);
        distTemp = distTemp ./ sum(distTemp);  % normalize distribution
        relDirDist(x,y,:) = distTemp;
    end
end
end
