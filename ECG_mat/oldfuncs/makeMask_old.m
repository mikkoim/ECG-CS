function mask = makeMask(s, locs, maskwidth)
mw = round(maskwidth/2);
N = length(s);

mask = zeros(N,1);

for l=locs
    
    if (l-mw) > 0 && (l+mw) < N
        mask(l-mw:l+mw) = 1;
    end
end

end