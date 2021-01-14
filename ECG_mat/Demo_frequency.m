clear all;
load('100_ECG_0_1800.mat','ECG_1');

% Parameters
N = 2^11;        % signal length
cr = 0.65; % compression ratio    
m = round(cr*N); % amount of samples

params.N = N;
params.cr = cr;
params.m = m;

% Preprocess signal
s = preprocess(ECG_1, params);

% transforms
Phi = make_Phi('dct', params);
PhiT = pinv(Phi);

% perform transform
x = PhiT*s;
plot(x);

%% Masking in frequency domain

% Parameters
params.em_power = 0.1; %embedding power
params.M = 90; % watermark length

% Create mask
params.start_i = 60;
params.end_i = 90;
mask = make_mask(x, 'fixed', params);
plot_comparison(x,mask,'Original signal and mask', 500)

% Perturbation matrix
params.p = 0.5; % Probability of corruption
MMatrix = make_perturbation_matrix(mask, params);

% Corrupt in frequency domain and transform back to time
t = MMatrix*x; % Corrupt coefficients
plot_comparison(s,Phi*t, 'Original and perturbed signal');

%% Perturbing and compressive sensing without watermark

% Measurement
A0 = make_A(params);
A = A0*PhiT; % Measurement matrix with transform

H = A*Phi; % H = gaussian*Phi*PhiT = gaussian
y = A0*MMatrix*PhiT*s; % Measurement of the freq-domain corrupt signal

% Optimization to see whether we can recover corrupted signal
x_hat = optimize(H,y);
s_hat = Phi*x_hat;

plot_comparison(s, s_hat, 'Compressively sensed original and perturbed signal');

plot_comparison(MMatrix*x, x_hat, 'Frequency domain');

%% Multi-level masking, adding watermark

% Watermark encryption matrix B and its annihilator F
[B,F] = make_B(params);

% Watermarking 
params.bitdepth = log2(N);
params.maskwidth = params.end_i - params.start_i +1;
w = make_watermark(mask, MMatrix, 'fixed', params);
[bw, v] = embed_watermark(w, B, y, params);

y_w = y + bw; % Corrupted signal with additive noise 

%% USER A %%%%%%%%%%%%%

x_hatA = optimize(H, y_w);
s_hatA = Phi*x_hatA;

plot_comparison(s, s_hatA, 'User A');
print_psnr_mask(s, s_hatA, mask);

plot_comparison(MMatrix*x, x_hatA, 'Masked raw signal vs compressively sensed');

%% USER B %%%%%%%%%%%%%%

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
H_D = A*Phi*MMatrix_hat; % Perturbation matrix changes place in frequency domain masking 
y_hat = y_w - B*w_hat;

x_hatB = optimize(H_D, y_hat);
s_hatB = Phi*x_hatB;

plot_comparison(s, s_hatB, 'User B');

