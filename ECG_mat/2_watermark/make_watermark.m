function [w, ww] = make_watermark(mask, MMatrix, params)
M = params.M; % Length of outgoing watermark


if strcmp(params.masktype, 'fixed') || strcmp(params.masktype, 'peak')
    bitdepth = params.bitdepth;
    maskwidth = params.maskwidth;
    ww = watermark_fixed(mask, MMatrix, bitdepth, maskwidth);
    
elseif strcmp(params.masktype, 'stationary')
    ww = watermark_stationary(mask,MMatrix);
end


assert( length(ww) < M, sprintf("Watermark size too large, lengths: M: %i, w: %i",M,length(ww)));
w = zeros(M,1);
w(1:length(ww)) = ww; % Extent to max mask length with zeros


%w = redundancy_apply(w);
end

function [ww] = watermark_fixed(mask, MMatrix, bitdepth, maskwidth)
D = diag(MMatrix);

% Mask width info
w_width = double( dec2bin(maskwidth, bitdepth) ) - 48; 
w_width = w_width';

% Mask location info
mask_d = diff(mask);
rising = find(mask_d == 1); % Rising edge of the mask

w_i0 = double( dec2bin( rising, bitdepth) ) - 48; 
w_i = w_i0(:);

% Location amount info
N_locs = size(w_i0,1);
w_nloc =  double( dec2bin(N_locs, bitdepth) ) - 48;
w_nloc = w_nloc';

% Mask info
w_m = D(mask == 1); % Corruption info
w_m(w_m == -1) = 0;

assert(length(w_m) == N_locs*maskwidth, 'Mask and locations do not match')

ww = [w_width; w_nloc; w_i; w_m];
ww(ww==0) = -1;
end

function ww = watermark_stationary(mask, MMatrix)
D = diag(MMatrix);

w_m = D(mask == 1); % Corruption info
ww = w_m;
end
