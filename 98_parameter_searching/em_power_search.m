clear all;
load('..\97_dataset_creation\mitdb_sample.mat')

% Fixed parameters
params.N = 2^11;                % signal length
params.cr = 0.3;               % compression ratio    

params.transform = 'bior4.4'; %bior4.4, dct, dct_mat
params.domain = 'time'; %frequency, time
params.masktype = 'peak'; %fixed, peak, stationary
params.maskwidth = 15; % Only for peak
%params.range = [20, 90]; %Only for fixed
params.precalc = false;

% Variable parameters-
em_powers = [2.5 3 3.5 4 4.5];
params.M = 500;
N_samples = 20;

ERRORS = zeros(N_samples, length(em_powers));
WWLEN = zeros(N_samples, length(em_powers));
A_error = zeros(N_samples, length(em_powers));
B_error = zeros(N_samples, length(em_powers));

sample_inds = randsample(size(X,1), N_samples);
for si = 1:N_samples
    f = sample_inds(si);
    s = preprocess(X(f,:), params);

    for i = 1:length(em_powers)
        fprintf(sprintf("sample %i, em power %.2f\n",si,em_powers(i)));
        params2 = params;
        params2.em_power = em_powers(i);          % embedding power

        try
            [s_hatA, s_hatB, errors, matrices, debug] = ECG_CS(s, params2);
        catch ME
            [s_hatA, s_hatB, errors, matrices, debug] = ECG_CS(s, params2);
        end
        ww_len = length(debug.ww);
        PSNR_A = psnr(s, s_hatA);
        PSNR_B = psnr(s, s_hatB);

        ERRORS(si,i) = errors;
        WWLEN(si,i) = ww_len;
        A_error(si,i) = PSNR_A;
        B_error(si,i) = PSNR_B;

    end
    
end

clear s s_hatA s_hatB X y