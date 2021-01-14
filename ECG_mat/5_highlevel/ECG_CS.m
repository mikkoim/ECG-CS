function [s_hatA, s_hatB, errors, matrices, debug] = ECG_CS(s, params)

params.m = round(params.cr*params.N);    % amount of samples

masktype = params.masktype;
domain = params.domain;
transform = params.transform;
precalc = params.precalc;

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

[A, B, F, Phi, y_w, v, debug_t] = transmitter_ecg(s, transform, mask, domain, precalc, params);

s_hatA = userA_ecg(y_w, A, Phi);

[s_hatB, debug_B] = userB_ecg(y_w, A, B, F, Phi, v, domain, params);

errors = sum(diag(debug_t.MMatrix) ~= diag(debug_B.MMatrix_hat));

matrices.A = A;
matrices.B = B;
matrices.F = F;
matrices.Phi = Phi;

debug.MMatrix = debug_t.MMatrix;
debug.MMatrix_hat = debug_B.MMatrix_hat;
debug.w = debug_t.w;
debug.ww = debug_t.ww;
debug.y_w = y_w;
end

