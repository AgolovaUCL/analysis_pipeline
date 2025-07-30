addpath('C:\Users\Sophia\Downloads\OScore_Matlab');



% Example values for the parameters
SamplingFrequency = 30000.0;     % Suppose we have a sampling frequency of 1 Khz.
TrialLength = round(SamplingFrequency * 2556.041967);             % Suppose we have a trial length of 4000 ms.
TrialNumber = 1;               % Suppose we have 20 trials; 
                                % IMPORTANT: if you have a single trial, the confidence score is not defined, hence it will be set to 0!

FMin = 5;                      % Suppose we want to look in the gamma-low band 30-50 Hz
FMax = 20;                      % Suppose we want to look in the gamma-low band 30-50 Hz
TrialList{1} = spikeTS;

disp([min(spikeTS) max(spikeTS) TrialLength]);

% Compute the oscillation score and confidence of the estimate
% The OScoreSpikes function requires 5 parameters:
%   1. TrialList - cell array of size (1 x Trial_Count) where each cell contains an array of spike times corresponding to one trial.
%   2. TrialLength - duration of trial in sample units
%   3. FMin - low boundary of the frequency band of interest in Hz
%   4. FMax - high boundary of the frequency band of interest in Hz
%   5. SamplingFrequency - sampling frequency of the time stamps in Hz
% and returns:
%   OS - oscillation score for the specified frequency band
%   CS - the confidence of the oscillation score estimate
%   OFq - peak oscillating frequency in the specified frequency band in Hz
%   AC - array containing the autocorrelogram computed on all trials
%   ACWP - array containing the autocorrelogram computed on all trials, smoothed and with no central peak
%   S - array containing the frequency spectrum of the smoothed peakless autocorrelogram

[OS, CS, OFq, AC, ACWP, S] = OScoreSpikes(TrialList, TrialLength, FMin, FMax, SamplingFrequency);

% NOTE: calling the function with less output variables, will return only the first outputs.
% Thus calling OS = OScoreSpikes(...) will return only the oscillation score, 
% and calling [OS, CS, OFq] = OScoreSpikes(...) will return the first 3 outputs.

% Plot the autocorrelogram and the smoothed, peakless autocorrelogram
CorrelationWindow = floor(size(AC,2)/2);
t=-CorrelationWindow:1:CorrelationWindow;
figure(1);
plot(t,AC);
xlabel('Time lag [bins]','FontSize',14);
ylabel('Count','FontSize',14);
title('Autocorrelogram (AC)','FontSize',14);
figure(2);
plot(t,ACWP);
xlabel('Time lag [bins]','FontSize',14);
ylabel('Count','FontSize',14);
title('Smoothed, peakless AC','FontSize',14);

% Plot the spectrum of the smoothed, peakless autocorrelogram
N = size(S,2);
f=0:SamplingFrequency/(2*N):SamplingFrequency/2*(N-1)/N;
figure(3);
plot(f,S);
xlabel('Frequency [Hz]','FontSize',14);
ylabel('Magnitude','FontSize',14);
title('Spectrum of the smoothed, peakless AC','FontSize',14);
