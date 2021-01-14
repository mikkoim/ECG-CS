function [s_hatB, infor] = userB_ECGF(y_w, ...
                            watermark_inf, ...
                            smean, ...
                            omegaA, ...
                            omegaF, ...
                            param)

%%%%%%%%%%%%%%%
% This function takes the compressed + encripted signal, and the parameters
% of the framework as the input, and it reconstrcts the signal s for User-B
%%%%%%%%%%%%%%%

S1 = param.S1; % Image dimensions.
S2 = param.S2;
N = S1*S2; % Total signal size per channel.
m = round(param.N*param.mratio); % Compressed signal size per channel.

% Transforms
h = MakeONFilter('Coiflet',2);
Wav = @(t) FWT2_POE(t,3,h); % Wavelet coeefficients of image.
inWav = @(t) IWT2_POE(t,3,h);
Wav1 = @(t) wavelet(t,Wav,S1,S2);
inWav1 = @(t) inwavelet(t,inWav,S1,S2);

% Measurements
% Encoding matrix (Measurement matrix) for the signal s.
A = @(t) Noiselet([t;zeros(param.redundant,1)],omegaA);
H  = @(t) Noiselet_inW(A,inWav1,t);
HT = @(t)  Adj_Noiselet(t,param.N,omegaA);

% Create user B key
temp = ones(m, 1);
temp(omegaF) = 0;
omegaB = find(temp == 1);

% Create transformation functions
B = @(t) At_DHT(t, omegaB, m);
BT = @(t) DHT(t, omegaB);

F   = @(t) DHT(t, omegaF);
FT  = @(t)  At_DHT(t, omegaF, m );

FH  = @(t) DHT(H(t),omegaF);
FHT = @(t)  Adj_Noiselet_inW(HT,Wav1,FT(t),N);

%%%%  Decoding Part %%%%%%%%
% Regularization parameter
tau = 4;
% Set tolA
tolA = 1.e-7;

y_tild = F(y_w);
% First estimation of x:
[~,x_tild,~,~,~,~]= ...
    GPSR_BB(y_tild,FH,tau,...
    'Debias',1,...
    'AT',FHT,...
    'Initialization',0,...
    'StopCriterion',1,...
    'ToleranceA',tolA,'ToleranceD',0.00001);

%%%%%%%%%%%%%%%%% Reconstruct watermark %%%%%
v = watermark_inf.v;
new_y = y_w - H(x_tild);
w_pp = BT(new_y);

[w_h, D_hat] = reconstruct_watermark(w_pp, v, param);

infor.total_error = sum(sum(D_hat ~= watermark_inf.D));

% Reconstruct original signal using matrix (A+M)
outside =  watermark_inf.outside;

A_D = @(t) A(outside(:).*t+D_hat(:).*t);
AT_Dt = @(t) outside(:).*new_A_T(t,HT,N) +D_hat(:).*new_A_T(t,HT,N);

A_D   = @(t) Noiselet_inW(A_D,inWav1,t);
AT_Dt = @(t)  Adj_Noiselet_inW(AT_Dt,Wav1,t,N);

% Regularization parameter
tau = 4;
% Set tolA
tolA = 1.e-5;
% Final estimation of x to form s,

newy2 = y_w - B(w_h);
[~,x_debias,~,~,~,~]= ...
    GPSR_BB(newy2,A_D,tau,...
    'Debias',1,...
    'AT',AT_Dt,...
    'Initialization',2,...
    'StopCriterion',1,...
    'ToleranceA',tolA,'ToleranceD',0.0001);

s_hat_h = inWav1(x_debias); % Inverse wavelet to compute s from x.
s_hat_h = s_hat_h + smean;
s_hatB = reshape(s_hat_h,S1,S2);

end

function out = new_A_T(t,phi_T,N)
s_hat1=phi_T(t);
out = s_hat1(1:N);
end