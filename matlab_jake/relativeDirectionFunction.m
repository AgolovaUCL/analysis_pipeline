function mrlData = relativeDirectionFunction(spikePos, spikeHD, ...
    spikePlats, relDirDists, angleEdges, frameSize)
%% RELATIVEDIRECTIONFUNCTION
% This function computes the mean resultant length (MRL) of relative head 
% directions of spikes with respect to goal platforms. It uses spike 
% positions, head directions, and precomputed distance distributions to 
% normalize directional histograms. The output is a struct with summary
% statistics about the directional tuning.

% Inputs:
%   spikePos, spikeHD, and spikePlats - can be foundin main consink function
%   relDirDist - function for this found
%   angleEdges - 0 to 360 in binsize steps
%   frameSize  - size of our x and y limits

% Convert bin edges into bin centers in radians
histBinCentres = deg2rad((angleEdges(2)-angleEdges(1))/2 + angleEdges(1:end-1));

% Define spatial binning (15 px resolution grid across the arena)
binSize = 15;
xAxis = 1:binSize:frameSize(1);
yAxis = 1:binSize:frameSize(2);

%% Combine relative distance distributions across platforms
totalDist_Rel = [];
for p = 1:length(relDirDists)
    % Number of spikes assigned to platform p
    nSpikesPerPlatform = length(find(spikePlats == p));
    if nSpikesPerPlatform == 0
        continue
    end
    
    % Weight the relative distributions by number of spikes
    platDist_Rel = relDirDists{p} * nSpikesPerPlatform;
    
    % Accumulate into total distributions
    if isempty(totalDist_Rel)
        totalDist_Rel = platDist_Rel;
    else
        totalDist_Rel = totalDist_Rel + platDist_Rel;
    end
end

% Reorder dimensions so bins correspond to [angle, y, x]
totalDist_permute = permute(totalDist_Rel, [3, 2, 1]);

%% Compute relative-to-goal directional histograms of spikes
dirRel2Goal_histCounts = dirHistCounts(spikePos, spikeHD, xAxis, yAxis, ...
    angleEdges, histBinCentres);

% Normalize by expected distribution
normDist = dirRel2Goal_histCounts ./ totalDist_permute;

% Scale to match total number of spikes
sumNormDist = sum(normDist, 1);
normDistFactor = length(spikeHD) ./ sumNormDist;
normDistFactor = repmat(normDistFactor, length(histBinCentres), 1, 1);
normDist = normDist .* normDistFactor;

% Compute MRL statistics from normalized distribution
mrlData.norm = mrlRelDir(normDist, xAxis, yAxis, histBinCentres);

end

%% ------------------------------------------------------------------------
function histCounts = dirHistCounts(pos, hd, xAxis, yAxis, angleEdges, ...
    histBinCentres)
% DIRHISTCOUNTS
% Computes directional histograms of relative head direction to spatial bins.
%
% Inputs:
%   pos         - spike positions (N x 2)
%   hd          - spike head directions (N x 1, degrees)
%   xAxis, yAxis- spatial bin coordinates
%   angleEdges  - angular bin edges for histogram
%   histBinCentres - centers of angular bins
%
% Output:
%   histCounts  - histogram of relative directions for each bin

% Distances from each spike to bin centers
xDistance = (xAxis - pos(:,1))';
yDistance = (yAxis - pos(:,2))';

if length(xAxis) > 1
    % Expand distances into 3D arrays: [x, y, spikes]
    xDistance = reshape(xDistance, [length(xAxis), 1, length(hd)]);
    xDistance = repmat(xDistance, 1, length(yAxis), 1);
    
    yDistance = reshape(yDistance, [1, length(yAxis), length(hd)]);
    yDistance = repmat(yDistance, length(xAxis), 1, 1);
end

% Direction from spike to bin (goal direction)
dir2goal = rad2deg(atan2(xDistance, yDistance));
dir2goal(dir2goal < 0) = dir2goal(dir2goal < 0) + 360;

if length(xAxis) == 1
    % Case: only 1 bin → simple 1D histogram
    dirRel2Goal = hd - dir2goal';
    dirRel2Goal(dirRel2Goal < 0) = 360 + dirRel2Goal(dirRel2Goal < 0);
    histCounts = histcounts(dirRel2Goal, angleEdges);
else
    % Case: multiple bins → compute histograms for each [x,y] bin
    spikeHD_XP = reshape(hd, 1, 1, length(hd));
    spikeHD_XP = repmat(spikeHD_XP, length(xAxis), length(yAxis));
    dirRel2Goal = spikeHD_XP - dir2goal;
    dirRel2Goal = permute(dirRel2Goal, [3, 2, 1]);
    
    dirRel2Goal(dirRel2Goal < 0) = 360 + dirRel2Goal(dirRel2Goal < 0);
    
    % Initialize histogram container
    histCounts = NaN(length(histBinCentres), size(dirRel2Goal, 2), size(dirRel2Goal, 3));
    for x = 1:length(xAxis)
        for y = 1:length(yAxis)
            histCounts(:, y, x) = histcounts(dirRel2Goal(:, y, x), angleEdges);
        end
    end
end
end

%% ------------------------------------------------------------------------
function mrlData = mrlRelDir(histCounts, xAxis, yAxis, histBinCentres)
% MRLRELDIR
% Computes MRL statistics (mean resultant length, preferred direction, etc.)
% from relative direction histograms.
%
% Inputs:
%   histCounts     - angular histograms [angles x y x]
%   xAxis, yAxis   - spatial bins
%   histBinCentres - bin centers in radians
%
% Output:
%   mrlData struct with fields:
%       mrl, dir, minDir, dirCoor, coor, pval, z, distribution,
%       allMRL, allDir

% Replicate angular bin centers to match histogram dimensions
histBinCenRep = repmat(histBinCentres', 1, length(yAxis), length(xAxis));

% Compute mean resultant length (MRL) across bins
mrl = circ_r(histBinCenRep, histCounts);
mrl = squeeze(mrl);
mrl_Lin = mrl(:);

% Find bin with maximal MRL
[mrl_Max, mrlInd] = max(mrl_Lin);
[mrl_MaxCoor(1), mrl_MaxCoor(2)] = ind2sub(size(mrl), mrlInd);
mrlData.mrl = mrl_Max;

% Preferred direction at maximal MRL bin
direction = circ_mean(histBinCenRep, histCounts);
direction = squeeze(direction);
mrlData.dir = rad2deg(direction(mrl_MaxCoor(1), mrl_MaxCoor(2)));

% Find minimum angular difference (closest to zero)
dir_Lin = direction(:);
[dir_Min, dirInd] = min(abs(dir_Lin));
[dir_MinCoor(1), dir_MinCoor(2)] = ind2sub(size(direction), dirInd);
mrlData.minDir = dir_Min;
dir_MinCoor(1) = yAxis(dir_MinCoor(1));
dir_MinCoor(2) = xAxis(dir_MinCoor(2));
mrlData.dirCoor = fliplr(dir_MinCoor);

% Perform Rayleigh test of non-uniformity at preferred bin
[pval, z] = circ_rtest(histBinCentres', histCounts(:, mrl_MaxCoor(1), mrl_MaxCoor(2)));
mrlData.distribution = squeeze(histCounts(:, mrl_MaxCoor(1), mrl_MaxCoor(2)))';
mrl_MaxCoor(1) = yAxis(mrl_MaxCoor(1));
mrl_MaxCoor(2) = xAxis(mrl_MaxCoor(2));
mrlData.coor = fliplr(mrl_MaxCoor);
mrlData.pval = pval;
mrlData.z = z;

% Store all MRL and direction maps for inspection
mrlData.allMRL = mrl;
mrlData.allDir = direction;

end
