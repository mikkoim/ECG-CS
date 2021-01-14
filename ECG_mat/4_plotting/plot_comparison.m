function plot_comparison(s, s_hat, title_string, plotmax)

assert(length(s) == length(s_hat),'Signals must be the same length');

if nargin < 4
    pm = length(s);
else
    pm= plotmax;
end

figure;
plot(s(1:pm)); hold on; plot(s_hat(1:pm))
title(title_string);

fprintf("PSNR: %.4f\n", psnr(s_hat,s))
fprintf("PDR: %.4f\n", norm(s_hat-s)./norm(s))
end

