fnames = {'frequency_fix_cr030.mat', 
    'frequency_fix_cr050.mat',
    'frequency_fix_cr065.mat',
    'frequency_stat_cr030.mat',
    'frequency_stat_cr050.mat',
    'frequency_stat_cr065.mat'};

figure; hold on
for i = 1:6
    fname = fnames{i};
    load(fname);
    dh = mean_freq_diff(S,SB);
    d.(fname(1:end-4)) = dh;
    plot(log(dh));
end

legend({'Freq, MR 0.3',
    'Freq, MR 0.5',
    'Freq, MR 0.65',
    'Fixed freq, MR 0.3',
    'Fixed freq, MR 0.5',
    'Fixed freq, MR 0.65',});

ylabel('log absolute distance')
xlabel('DCT component')
