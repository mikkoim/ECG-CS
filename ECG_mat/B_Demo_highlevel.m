clear all;
load('100_ECG_0_1800.mat','ECG_1');

% Parameters

params.N = 2^11;                % signal length
params.cr = 0.65;               % compression ratio    
params.m = round(params.cr*params.N);    % amount of samples
params.em_power = 0.05;          % embedding power
params.M = 80;                 % watermark length

% Preprocess signal
s = preprocess(ECG_1, params);

% Peak mask
if 0
    [~, ~, ~, R_i, S_i, ~]  = rpeakdetect(s,1,0.5,0);
    params.maskwidth = 15;
    params.locs = R_i;
    mask = make_mask(s, 'peaks', params);
    
else % Fixed mask
    params.start_i = 20;
    params.end_i = 90;
    params.maskwidth = params.end_i - params.start_i +1;
    mask = make_mask(s, 'fixed', params);
end

plot_comparison(s,mask,'Original signal and mask')

%% Transmitter
transform = 'dct_mat';
domain = 'time';
precalc = false;
[A, B, F, Phi, y_w, v, debug_t] = transmitter_ecg(s, transform, mask, domain, precalc, params);

%% USER A %%%%%%%%%%%%%
s_hatA = userA_ecg(y_w, A, Phi);
plot_comparison(s, s_hatA, 'User A');
print_psnr_mask(s, s_hatA, mask);

%% USER B %%%%%%%%%%%%%%%
[s_hatB, debug_B] = userB_ecg(y_w, A, B, F, Phi, v, domain, params);

plot_comparison(diag(debug_t.MMatrix), diag(debug_B.MMatrix_hat), ...
    sprintf("Errors in mask recovery: %i",sum(diag(debug_t.MMatrix) ~= diag(debug_B.MMatrix_hat))), 500);

plot_comparison(s, s_hatB, 'User B');
