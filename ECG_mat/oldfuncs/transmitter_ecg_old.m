function [y_w, watermark_inf, A, omegaF, smean] = transmitter_ecg(S0, mask, param)

%%%%%%%%%%%%%%%
% This function takes the input image, mask for the privacy sensitive parts
% and the required parameters for the algorith, and it produces encripted
% and compressed signal as the output: y_w = y_d + Bw = (A + M)s + Bw.
%%%%%%%%%%%%%%%

M = param.M;
N = param.S1 * param.S2;
m = round(param.mratio * param.N); % Number of measurements for CS (each channel)

%%% Create watermark
bitdepth = log2(N);
[w, watermark_inf] = watermark(mask, M, bitdepth);
outside = watermark_inf.outside;
D = watermark_inf.D;

% Prepare signal
S = double(S0);
smean = mean(S); % Mean normalization.
S = S - smean;

% Encoding matrix for the signal s
A = sqrt(1/m)*randn(m,N);
A = orth(A')';

% Corrupted encoding matrix for the signal s: (A + M)
t = outside(:).*s + D(:).*s;

% Construct Encoding for the watermark.
T = M;

B = sqrt(1/m)*randn(m,T);
B = orth(B')';
BT = pinv(B);

F = null(B','r')';
F = orth(F')';

% Take measurements from the signal for each color channel;
y = A*t;

bw = B*w; % Encoding the watermark, B*w
bw = bw./norm(bw); % Normalization

% Embedding power = Bw/y_d for ensuring good reconstruction quality for the User-A.
alpha1 = norm(y).*(param.em_power);
bw = bw.*alpha1;

watermark_inf.v = check_watermark(bw,omegaB,w);

y_w = y + bw; % Obtaining final compressed and encrypted signal, y_w = y + bw

end

function v =check_watermark(bw,in,www)
% Check watermark
bw_in = DHT(bw,in);
v = abs(bw_in(1));

w = v*www;

w_h=zeros(size(bw_in));
w_h(bw_in >= 0.1) = v*1;
w_h(bw_in <- 0.1) = v*-1;

error_in= (sum((w_h-w)>10^-1));

if error_in ~=0
    error('something wrong in watermark')
end

end