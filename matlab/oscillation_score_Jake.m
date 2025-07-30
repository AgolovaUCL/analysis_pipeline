
%spikeTS = spikeTS/30;

oscScore = OscillationScore(spikeTS, 5, 20, 2556);

function [oscScore] = OscillationScore(spikes, fmin, fmax, durationSec)
% from The oscillation score: an efficient method for estimating 
% oscillation strength in neuronal activity, Muresan et al. 2008
%%
oscScore.binSize = 1; % in ms
sFreq = oscScore.binSize * 1000;
oscScore.sFreq = 1000/oscScore.binSize; % in Hz
doubleWSize = 2^(floor(max(log2((3*sFreq)/fmin), log2(sFreq/4))) + 1);
windowSize = doubleWSize/2;


%%
nSpikes = length(spikes);
spikesSec = spikes / 1000;
nDivisions = doubleWSize / oscScore.binSize;
[bins, counts] = interspike_histogram(spikesSec, spikesSec, doubleWSize, ...
    'divisions', nDivisions, ...
    'trialDur', durationSec, ...
    'plot', 1);

counts = counts/durationSec;
oscScore.autocorr.singleSided = counts;


figure;
bar(bins, counts, 'k');
xlabel('Lag (ms)');
ylabel('Count');
title('Autocorrelogram');


mirrorCounts = [fliplr(counts) nSpikes/durationSec counts];
rawMidInd = ((length(mirrorCounts) -1)/2)+1; 
mirrorTrunc = mirrorCounts(rawMidInd - windowSize: rawMidInd + windowSize);
oscScore.autocorr.raw = mirrorTrunc;

% subplot(3,2,1)
% bar(mirrorTrunc)
%%
sdFast = min(2, 134/(1.5*fmax)) * (oscScore.sFreq/1000);
gaussKFast = (1/(sdFast * sqrt(2*pi))) * exp((-1/2)*(((-windowSize:windowSize)./sdFast).^2));
gaussKFast = gaussKFast/sum(gaussKFast);
gaussKFast(gaussKFast < 1e-4) = [];  
gaussKFast = gaussKFast/sum(gaussKFast);

countsFastSmooth = conv(mirrorCounts, gaussKFast, 'valid');
fastInd = ((length(countsFastSmooth) -1)/2)+1; 
fastTrunc = countsFastSmooth(fastInd - windowSize: fastInd + windowSize);
oscScore.autocorr.fastSmooth = fastTrunc;

% subplot(3,2,3)
% bar(fastTrunc)
%%
sdSlow = 2 * (134/(1.5 * fmin)) * (oscScore.sFreq/1000);
gaussKSlow = (1/(sdSlow * sqrt(2*pi))) * exp((-1/2)*(((-windowSize:windowSize)./sdSlow).^2));
gaussKSlow = gaussKSlow/sum(gaussKSlow);
gaussKSlow(gaussKSlow < 1e-4) = [];  
gaussKSlow = gaussKSlow/sum(gaussKSlow);

countsSlowSmooth = conv(mirrorCounts, gaussKSlow, 'valid');
slowInd = ((length(countsSlowSmooth) -1)/2)+1; 
slowTrunc = countsSlowSmooth(slowInd - windowSize: slowInd + windowSize);
oscScore.autocorr.slow = slowTrunc;

% subplot(3,2,5)
% bar(slowTrunc)

scalingFactor = doubleWSize/countsSlowSmooth(slowInd);
slopeSlow = gradient(countsSlowSmooth) * scalingFactor;
angleSlow = atan(slopeSlow) * 180/pi;

% smallAngleInd = find(angleSlow(1:slowInd-1) < 10);
% smallAngleInd = smallAngleInd(end);

smallAngleInd = find(angleSlow(slowInd - 101:slowInd-1) < 10);
if isempty(smallAngleInd)
    [~, peakInd] = max(slopeSlow);    
    [~, smallAngleInd] = min(slopeSlow(slowInd-101:peakInd));
    smallAngleInd = smallAngleInd + slowInd - 102;
else
    smallAngleInd = smallAngleInd(end) + (slowInd - 102);
end
%%
lengthDiffFastSlow = length(countsFastSmooth) - length(countsSlowSmooth);
countsNoPeak = countsFastSmooth(1:fastInd);
countsNoPeak(smallAngleInd + (lengthDiffFastSlow/2):end) = ...
    countsNoPeak(smallAngleInd + lengthDiffFastSlow/2);

countsNoPeakTrunc = [countsNoPeak fliplr(countsNoPeak(1:end-1))];
countsNoPeakTrunc = countsNoPeakTrunc(fastInd - windowSize: fastInd + windowSize);
oscScore.autocorr.noPeak = countsNoPeakTrunc;

% subplot(3,2,2)
% bar(countsNoPeakTrunc)
%% compute wavelet

% subplot(3,2,4)
[wt,f] = cwt(countsNoPeakTrunc,oscScore.sFreq, 'VoicesPerOctave', 48);
wavPower = abs(wt);
% imagesc(wavPower)
oscScore.wave.power = wavPower;
oscScore.wave.freq = f;

nBins2Remove = length(countsNoPeak) - (smallAngleInd + (lengthDiffFastSlow/2));
wavPowerCentInd = (length(countsNoPeakTrunc) - 1)/2 + 1;
wavPowerInd = [1:wavPowerCentInd-75, wavPowerCentInd+75:length(countsNoPeakTrunc)];
avgSpectrum = mean(wavPower(:, wavPowerInd), 2);
freqBand = f(f>= fmin & f <= fmax);
% freqBand = f(f>= 7 & f <= fmax);
spectrumBand = avgSpectrum(f>= fmin & f <= fmax);
% spectrumBand = avgSpectrum(f>= 7 & f <= fmax);
oscScore.wave.avgSpectrum = avgSpectrum;
oscScore.wave.freqBand = freqBand;
oscScore.wave.spectrumBand = spectrumBand;

% subplot(3,2,6)
% plot(freqBand, spectrumBand)

% pks = findpeaks(spectrumBand);
% pkFreq = freqBand(pks.loc);
% pkFreqThetaDiff = abs(8.2 - pkFreq);
% [~, minDiffInd] = min(pkFreqThetaDiff);
% peakInd = pks.loc(minDiffInd);

[pks,locs]= findpeaks(spectrumBand);
pkFreq = freqBand(locs);
pkFreqThetaDiff = abs(8.2 - pkFreq);
[~, minDiffInd] = min(pkFreqThetaDiff);
peakInd = locs(minDiffInd);

oscFreq = freqBand(peakInd);
peakPower = spectrumBand(peakInd);   
if nSpikes < 20
    oscScore.oscFreq = NaN;
else
    oscScore.oscFreq = oscFreq;
end

intgrlPower = trapz(flipud(f),flipud(avgSpectrum))./(f(1) - f(end)); % I verified that this does indeed provide an average power by performing the computation on a small area around the osc freq
%%
if length(peakPower) > 1
    oscScore.value = NaN;
else
    oscScoreTemp = peakPower/intgrlPower;
    oscScore.value = oscScoreTemp;
end


end




