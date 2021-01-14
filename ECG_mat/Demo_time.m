clear all;
load('100_ECG_0_1800.mat','ECG_1');

% Parameters
N = 2^11;           % signal length
cr = 0.65;          % compression ratio    
m = round(cr*N);    % amount of samples

params.N = N;
params.cr = cr;
params.m = m;

% Preprocess signal
s = preprocess(ECG_1, params);

% Peak detection
[~, ~, ~, R_i, S_i, ~]  = rpeakdetect(s,1,0.5,0);
figure;
plot(s); hold on; scatter(R_i, s(R_i)); scatter(S_i, s(S_i));

% transforms
Phi = make_Phi('bior4.4', params);
PhiT = pinv(Phi);

% measurements
A = make_A(params);
H = A*Phi;
y = A*s;

% Optimization
x_hat = optimize(H,y);
s_hat = Phi*x_hat;

plot_comparison(s, s_hat, 'Basic compressive sensing, no masking');

%% Partial masking

% Parameters
params.em_power = 1;     % embedding power
params.M = 300;              % watermark length

% Create mask
params.maskwidth = 15;
params.locs = R_i;
mask = make_mask(s, 'peaks', params);
plot_comparison(s,mask,'Original signal and mask')

% Perturbation matrix
params.p = 0.5; % Probability of corruption
MMatrix = make_perturbation_matrix(mask, params);

% Watermark encryption matrix B and its annihilator F
[B,F] = make_B(params);

% Watermarking 
params.bitdepth = log2(N);
w = make_watermark(mask, MMatrix, 'fixed', params);
[bw, v] = embed_watermark(w, B, y, params);

% Masked measurements
t = MMatrix*s; % Corrupt signal with masking matrix M
plot_comparison(t,s,'Corrupted signal and original');

y = A*t;
y_w = y + bw; % Corrupted signal with additive noise 

%% USER A %%%%%%%%%%%%%

x_hatA = optimize(H, y_w);
s_hatA = Phi*x_hatA;

plot_comparison(s, s_hatA, 'User A');
print_psnr_mask(s, s_hatA, mask);

plot_comparison(MMatrix*s, s_hatA, 'Masked raw signal vs compressively sensed');


%% USER B %%%%%%%%%%%%%%%

% Watermark recovering
FH = F*H;
BT = pinv(B);
y_tild = F*y_w;
x_tild = optimize(FH, y_tild);

w_pp = BT*(y_w - H*x_tild);
[w_hat, MMatrix_hat] = recover_watermark(w_pp, v, N);

plot_comparison(diag(MMatrix), diag(MMatrix_hat), ...
    sprintf("Errors in mask recovery: %i",sum(diag(MMatrix) ~= diag(MMatrix_hat))), 500);

% Original signal decoding
H_D = A*MMatrix_hat*Phi; 
y_hat = y_w - B*w_hat;

x_hatB = optimize(H_D, y_hat);
s_hatB = Phi*x_hatB;

plot_comparison(s, s_hatB, 'User B');
