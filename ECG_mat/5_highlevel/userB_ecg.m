function [s_hatB, debug] = userB_ecg(y_w, A, B, F, Phi, v, domain, params)
H = A*Phi;
N = params.N;

% Watermark recovering
FH = F*H;
BT = pinv(B);
y_tild = F*y_w;
x_tild = optimize(FH, y_tild);

w_pp = BT*(y_w - H*x_tild);
[w_hat, MMatrix_hat] = recover_watermark(w_pp, v, params);

% Original signal decoding
if strcmp(domain,'time')
    H_D = A*MMatrix_hat*Phi; 
elseif strcmp(domain, 'frequency')
    H_D = A*Phi*MMatrix_hat; 
end

y_hat = y_w - B*w_hat;

x_hatB = optimize(H_D, y_hat);
s_hatB = Phi*x_hatB;

%% Debug variables
debug.MMatrix_hat = MMatrix_hat;
end

