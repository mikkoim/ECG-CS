function print_psnr_mask(s, s_hat, mask)

outside = (mask-1)*-1; 

fprintf("PSNR: %.4f\n", psnr(s_hat,s))
fprintf("PSNR inside: %.4f\n", psnr(s_hat(logical(mask)),s(logical(mask))))
fprintf("PSNR outside: %.4f\n", psnr(s_hat(logical(outside)),s(logical(outside))))
end

