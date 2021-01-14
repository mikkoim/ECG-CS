function [s_hatA] = userA_ecg(y_w, A, Phi)

H = A*Phi;
x_hatA = optimize(H, y_w);
s_hatA = Phi*x_hatA;

end

