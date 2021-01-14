function [x_hat] = optimize(H, y)

x_init = pinv(H)*y ; % minimum norm solution
x_hat=l1eq_pd(x_init, H, [], y, 1e-3);

end

