function [A, B, F, Phi, y_w, v, debug] = transmitter_ecg(s, transform, mask, domain, precalc, params)

if precalc
    Phi = params.Phi; % Transform
    A0 = params.A; % Measurement
    B = params.B;
    F = params.F;
else
    Phi = make_Phi(transform, params);
    A0 = make_A(params);
    % Watermark encryption matrix B and its annihilator F
    [B,F] = make_B(params);
end

if strcmp(domain,'time')
    A = A0;
elseif strcmp(domain, 'frequency')
    PhiT = pinv(Phi);
    A = A0*PhiT;
else
   error('Not a known domain'); 
end

%% Masking
% Perturbation matrix
params.p = 0.5; % Probability of corruption
MMatrix = make_perturbation_matrix(mask, params);

% Masked measurements
if strcmp(domain,'time')
    y = A*MMatrix*s; % Corrupt signal with masking matrix M
    
elseif strcmp(domain, 'frequency')
    y = A0*MMatrix*PhiT*s;
end

% Watermarking
params.bitdepth = log2(params.N);
[w, ww] = make_watermark(mask, MMatrix, params);
[bw, v] = embed_watermark(w, B, y, params);

y_w = y + bw; % Corrupted signal with additive noise 

%% Variables for debugging
debug.w = w;
debug.ww = ww;
debug.MMatrix = MMatrix;
debug.y = y;
end

