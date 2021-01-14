function [s_hatA] = userA_ECGF(y_w, smean, omegaA, param)

%%%%%%%%%%%%%%%
% This function takes the compressed + encripted signal, and the parameters
% of the framework as the input, and it reconstrcts the signal s for User-A
%%%%%%%%%%%%%%%

S1 = param.S1; % Image dimensions
S2 = param.S2;
N = S1*S2; % Total signal size per channel.

% Transforms (Phi)
h = MakeONFilter('Coiflet',2);
Wav = @(t) FWT2_POE(t,3,h); % Wavelet coeefficients of image.
inWav = @(t) IWT2_POE(t,3,h);
Wav1=@(t) wavelet(t,Wav,S1,S2);
inWav1=@(t) inwavelet(t,inWav,S1,S2);

% Measurements
A = @(t) Noiselet([t;zeros(param.redundant,1)],omegaA);
H = @(t) Noiselet_inW(A,inWav1,t); % H = Phi * A

% Adjoints of the above matrices
AT = @(t)  Adj_Noiselet(t,param.N,omegaA);
HT = @(t)  Adj_Noiselet_inW(AT,Wav1,t,N); 

%%%%   Decoding Part  %%%%%%%%
% Regularization parameter
tau = 4;
% Set tolA
tolA = 1.e-7;

y_tild = y_w;

[~,x_tild1,~,~,~,~]= ...
    GPSR_BB(y_tild,H,tau,...
    'Debias',1,...
    'AT',HT,...
    'Initialization',0,...
    'StopCriterion',1,...
    'ToleranceA',tolA,'ToleranceD',0.00001);

s_hat_h = inWav1(x_tild1); % Inverse wavelet to compute s from x.
s_hat_h = s_hat_h + smean;
s_hatA = reshape(s_hat_h, S1, S2);

end

