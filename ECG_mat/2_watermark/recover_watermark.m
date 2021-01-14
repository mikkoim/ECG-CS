function [w_hat, MMatrix_hat] = recover_watermark(w_pp, v, params)
N = params.N;
w_hat = zeros(size(w_pp));
w_hat(w_pp >= 0) = v*1;
w_hat(w_pp < 0) = v*-1;

d = zeros(size(w_hat));
d(w_hat > 0) = 1;
d(w_hat < 0) = 0;

%d = redundancy_reduce(d);
try
    [D_hat, mask_hat, wm_length] = recover_wm(d, params);
    w_hat(wm_length:end) = 0;
    outside = (mask_hat-1)*-1;

    MMatrix_hat = diag(outside + D_hat);
catch ME
    warning(['Watermark not recovered properly: ' ME.message]);
    MMatrix_hat = eye(N);
end
end

function [D_hat, mask_hat, wm_length] = recover_wm(d, params)

if strcmp(params.masktype, 'fixed') || strcmp(params.masktype, 'peak')
    
    [D_hat, mask_hat, wm_length] = recover_fixed_wm(d, params.N);
    
elseif strcmp(params.masktype, 'stationary')
    
    [D_hat, mask_hat, wm_length] = recover_stationary_wm(d, params);
    
end

end

function [D_hat, mask_hat, wm_length] = recover_stationary_wm(d, params)
D = d;
D(D==0) = -1;

mask_hat = zeros(params.N,1);
mask_hat(params.start_i:params.end_i) = 1;

D_hat = zeros(params.N,1);
D_hat(mask_hat == 1) = D(1:params.maskwidth);

wm_length = params.maskwidth;

end

function [D_hat, mask_hat, wm_length] = recover_fixed_wm(d, N)

bd = log2(N);
to_dec = @(s) bin2dec(num2str(s));

w_width = to_dec(d(1:bd)');
w_nloc = to_dec(d(bd+1:2*bd)');

d_i = d(2*bd+1:2*bd+bd*w_nloc);
d_i = reshape(d_i,w_nloc, bd);
w_i = to_dec(d_i);

wm_length = 2*bd+bd*w_nloc + w_nloc*w_width;
w_m = d(2*bd+bd*w_nloc+1:wm_length);
w_m(w_m == 0) = -1;

% Recover mask from w_i

mask_hat = zeros(N,1);
for i = 1:length(w_i)
    loc = w_i(i);
    mask_hat(loc+1:loc+w_width) = 1;
end

D_hat = zeros(N,1);
D_hat(mask_hat==1) = w_m;

end
