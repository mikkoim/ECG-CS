clear all;
load('..\97_dataset_creation\mitdb_sample.mat')

% Fixed parameters
params.N = 2^11;                % signal length
params.cr = 0.3;               % compression ratio    

params.masktype = 'peak'; %fixed, peak, stationary
params.maskwidth = 15; % Only for peak
params.range = [20, 90]; %Only for fixed
params.precalc = false;
params.M = 1000;

ww_list = [];
for i = 1:size(X,1)
    s = preprocess(X(i,:), params);

    [mask, params] = masking(s,params);

    params.p = 0.5; % Probability of corruption
    params.bitdepth = log2(params.N);
    MMatrix = make_perturbation_matrix(mask, params);
    [w, ww] = make_watermark(mask, MMatrix, params);
    ww_list(i,1) = length(ww);
end

max(ww_list)


function [mask, params] = masking(s,params)
    masktype = params.masktype;
    if strcmp(masktype,'peak')
        [~, ~, ~, R_i, ~, ~]  = rpeakdetect(s,1,0.5,0);    
        fprintf(sprintf('%i peaks detected\n',length(R_i)));
        params.locs = R_i;
        mask = make_mask(s, 'peaks', params);

    elseif strcmp(masktype,'fixed') || strcmp(masktype,'stationary') 
        params.start_i = params.range(1);
        params.end_i = params.range(2);
        params.maskwidth = params.end_i - params.start_i +1;
        mask = make_mask(s, 'fixed', params);
    else
        error('Not a known mask type');
    end 
end
