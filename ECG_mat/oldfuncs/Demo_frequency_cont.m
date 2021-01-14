clear all;
load('100_ECG_0_1800.mat','ECG_1');

% Parameters
N = 2^11;        % signal length
CR = 0.65; % compression ratio    
m = round(CR*N); % amount of samples

% Preprocess signal
s = ECG_1(1:N);
s = (s-min(s));
s = s./max(s);

%%%%%%%%%transforms%%%%%%%%%%
Level = 5;
waveletName = "bior4.4";
Wav = @(t) wavedec(t,Level,waveletName);
%create transform matrix
I=eye(N);
for i=1:N
    PhiT(:,i)=dct(I(:,i));
    i
end

Phi=pinv(PhiT);

% Transform
x = PhiT*s; %original x
N_x = length(x);
figure,plot(x)

%% Masking in frequency domain
mask = zeros(N_x,1);
mask(30:60) = 1;

figure;
plot(mask(1:500)); hold on; plot(x(1:500));
title('Original coefficients and mask');

% Corruption matrix
p = 0.5; % Probability of corruption

MM = rand(N_x,1) < p; % Corrupt whole signal
MM = MM*2 -1; % Set to -1,1

D = mask.*MM;
outside = (mask-1)*-1; % Inverse of mask
MMatrix = diag(outside + D); % masking matrix M

t = MMatrix*x; % Corrupt coefficients
figure;
plot(x); hold on; plot(t);

% Inverse transform
s_hat = Phi*t;

figure;
plot(s); hold on; plot(s_hat)

disp(["PSNR: " psnr(s_hat,s)])
disp(["PDR: " norm(s_hat-s)./norm(s)])


%% Compressive sensing for masked signal

% Measurement
A0 = sqrt(1/m)*randn(m,N);
A0 = orth(A0')';
A = A0*PhiT; % Measurement matrix with transform

H = A*Phi; % H = gaussian*Phi*PhiT = gaussian
y = A0*MMatrix*PhiT*s; % Measurement of the freq-domain corrupt signal

% Optimization to see whether we can recover corrupted signal
x_init = pinv(H)*y;
x_hat=l1eq_pd(x_init, H, [], y, 1e-3);

s_hat = Phi*x_hat;

figure;
plot(s); hold on; plot(s_hat)
title('Time-domain');

figure;
plot(MMatrix*x); hold on; plot(x_hat)

disp(["PSNR: " psnr(s_hat,Phi*MMatrix*x)])
disp(["PDR: " norm(s_hat-s)./norm(s)])

%% Multi-level masking
em_power = 0.05; %embedding power
M = 50; % watermark length
% Watermark measurement matrix B
T = M;

B = sqrt(1/m)*randn(m,T);
BT = pinv(B);

F = null(B','r')'; % Annihilation matrix
F = orth(F')';

%%%%% Watermarking %%%%%%
w_m = D(mask==1); % Mask information
w = zeros(M,1); 
w(1:length(w_m)) = w_m; % Extent to max mask length with zeros

bw = B*w; % Embed watermark
bw = bw./norm(bw); % Normalization

alpha = norm(y).*(em_power);
bw = bw.*alpha;

bw_in = BT*bw; % Inverse embedding
v = abs(bw_in(1));

y_w = y + bw; % Corrupted signal with additive noise 

%% USER A %%%%%%%%%%%%%

%%% Optimization
x_init = pinv(H)*y_w;
x_hat=l1eq_pd(x_init, H, [], y_w, 1e-3);

s_hatA = Phi*x_hat;

figure;
plot(s); hold on; plot(s_hatA);
title('User A signal');

fprintf("PSNR A: %.4f\n", psnr(s_hatA,s))
fprintf("PDR A: %.4f\n", norm(s_hatA-s)./norm(s))

%% USER B %%%%%%%%%%%%%%

FH = F*H;

y_tild = F*y_w;

%%% Watermark recovering
x_init = pinv(FH)*y_tild;
x_tild = l1eq_pd(x_init, FH, [], y_tild, 1e-3);

w_pp = BT*(y_w - H*x_tild);
w_hat = zeros(length(w_pp),1);
w_hat(w_pp >= 0) = v*1;
w_hat(w_pp < 0) = v*-1;

D_hat = zeros(length(w_hat),1);
D_hat(w_hat > 0) = 1;
D_hat(w_hat < 0) = -1;

D2 = zeros(N,1);
D2(30:60) = D_hat(1:31); % Watermark position hardcoded in this example

w_hat(32:end) = 0;

MM2 = outside + D2;

figure;
plot(D2(1:500)); hold on; plot(D(1:500));
title(sprintf("Errors in mask recovery: %i",sum(D2 ~= D)))


%%%%%% Signal decoding %%%%%%%%%%%
MM2mat = diag(MM2); % Decoded corruption mask in matrix form
H_D = A*Phi*MM2mat; 

newy2 = y_w - B*w_hat;

%%% Optimization
x_init = pinv(H_D)*newy2;
x_hatB=l1eq_pd(x_init, H_D, [], newy2, 1e-3);

s_hatB = Phi*x_hatB;

figure;
plot(s); hold on; plot(s_hatB);
title('User B signal');

fprintf("PSNR B: %.4f\n", psnr(s_hatB,s))
fprintf("PDR B: %.4f\n", norm(s_hatB-s)./norm(s))
