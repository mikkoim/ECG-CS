function [mask] = make_mask(s, method, params)

if strcmp(method,'fixed')
    start_i = params.start_i;
    end_i = params.end_i;
    N = params.N;
    mask = zeros(N,1);
    mask(start_i:end_i) = 1; % Arbitary manually set mask
    
elseif strcmp(method,'peaks')
    % Remove close peaks
    too_close = [0 diff(params.locs) <= params.maskwidth];
    params.locs = params.locs(~too_close);
    params.locs(params.locs <= floor(params.maskwidth/2)) = [];
    
    mask = peak_mask(s, params.locs, params.maskwidth);
end

end

function mask = peak_mask(s, locs, maskwidth)
mw = floor(maskwidth/2);
N = length(s);

mask = zeros(N,1);

for l=locs
    
    if (l-mw) >= 0 && (l+mw) < N
        mask(l-mw:l+mw) = 1;
    end
end

end