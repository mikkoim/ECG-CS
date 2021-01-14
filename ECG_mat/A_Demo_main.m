clear all;
load('100_ECG_0_1800.mat','ECG_1');

% Parameters
params.N = 2^11;                % signal length
params.cr = 0.65;               % compression ratio    
params.em_power = 0.05;          % embedding power
params.M = 80;                % watermark length

params.transform = 'dct_mat'; %bior4.4, dct, dct_mat
params.domain = 'frequency'; %frequency, time
params.masktype = 'stationary'; %fixed, peak, stationary
params.precalc = false;
params.maskwidth = 15; % Only for peak
params.range = [20, 90]; %Only for fixed

s = preprocess(ECG_1, params);
[s_hatA, s_hatB, errors, matrices, debug] = ECG_CS(s, params);

plot_comparison(s, s_hatA, 'User A, embedding power 0.1');
plot_comparison(s, s_hatB, 'User B, embedding power 0.1');
