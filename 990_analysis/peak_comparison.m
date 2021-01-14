load('..\99_masking\MIT_masked\100_block_masked.mat')

e = info.errors;
for si = 1:128
    

    s = S(si,:)';
    sA = SB(si,:)';

    psnrs(si) = psnr(s,sA);

    [~, ~, ~, r, ~, ~]  = rpeakdetect(s,1,0.5,0);
    if e(si) ~= Inf
        [~, ~, ~, ra, ~, ~]  = rpeakdetect(sA,1,0.5,0);
    else
        ra = [];
    end

    [precision, recall] = match_peaks(r,ra);

    P(si,1) = precision;
    R(si,1) = recall;

end

mean(psnrs)

mean(P)
mean(R)

subplot(211)
plot(s); hold on; scatter(r,s(r),'go'); 
subplot(212)
plot(sA); hold on; scatter(ra,sA(ra),'r+');
