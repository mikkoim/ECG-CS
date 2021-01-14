clear all; clc;

folder = '\\kk11\e1007914$\Documents\SAMI\create_new_MIT_masked\MIT_block_ds';
d = dir(fullfile(folder,'*.mat'));

% Parameters
params.N = 2^11;                % signal length
params.cr = 0.65;               % compression ratio    
params.em_power = 0.1;          % embedding power
params.M = 70;                  % watermark length

params.transform = 'dct_mat'; %bior4.4, dct, dct_mat
params.domain = 'frequency'; %frequency, time
params.masktype = 'fixed'; %fixed, peak
params.precalc = false;
params.range = [60, 90];

for i = 2:length(d) % Go through each file separately
    load(fullfile(folder,d(i).name));

    S =  zeros(128,2048);
    SA = zeros(128,2048);
    SB = zeros(128,2048);
    err = ones(128,1);

    for ii = 1:size(ecg_block,1) % Go through each sample in the file
        fprintf(sprintf("\n\n\n\n %i \n\n\n\n",ii));
        s = preprocess(ecg_block(ii,:), params);
        try
            tic;
            [s_hatA, s_hatB, errors, matrices, debug] = ECG_CS(s, params); 
            toc;
        catch ME
            warning('Something went wrong');
            s_hatA = zeros(2048,1);
            s_hatB = zeros(2048,1);
            errors = inf;
        end
        err(ii) = errors;
        S(ii,:) = s;
        SA(ii,:) = s_hatA;
        SB(ii,:) = s_hatB;
    end
    info.params = params;
    info.errors = err;
    fname = [d(i).name(1:3) '_block_masked.mat'];
    save(fullfile('MIT_masked',fname), 'S', 'SA', 'SB', 'info');
end