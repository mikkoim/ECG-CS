clear all; clc;

load('..\97_dataset_creation\mitdb_sample.mat');

% Parameters
params.N = 2^11;                % signal length
params.cr = 0.3;               % compression ratio    
params.em_power = 4.5;          % embedding power
params.M = 500;                  % watermark length

params.transform = 'bior4.4'; %bior4.4, dct, dct_mat
params.domain = 'time'; %frequency, time
params.masktype = 'peak'; %fixed, peak, stationary
params.maskwidth = 15; % Only for peak
params.range = [20, 90]; %Only for fixed
params.precalc = false;

N_sig = size(X,1);

S =  zeros(N_sig,2048);
SA = zeros(N_sig,2048);
SB = zeros(N_sig,2048);
err = ones(N_sig,1);

parpool(4)
t = tic;
parfor i = 1:size(X,1)

    fprintf(sprintf("\n\n\n\n %i \n\n\n\n",i));
    s = preprocess(X(i,:), params);
    try
        %tic;
        [s_hatA, s_hatB, errors, matrices, debug] = ECG_CS(s, params); 
        %toc;
    catch ME
        warning(['Something went wrong: ' ME.message]);
        s_hatA = ones(2048,1);
        s_hatB = ones(2048,1);
        errors = inf;
    end
    
    err(i) = errors;
    S(i,:) = s;
    SA(i,:) = s_hatA;
    SB(i,:) = s_hatB;
end
toc(t)

info.params = params;
info.errors = err;
fname = [params.domain '_' datestr(datetime,'mmm-dd_HHMMSS') '_masked.mat'];
save(fname, 'X', 'S', 'SA', 'SB', 'y', 'info');