function [MMatrix] = make_perturbation_matrix(mask, params)

p = params.p;
N = params.N;

MM = rand(N,1) < p; % Corrupt whole signal
MM = MM*2 -1; % Set to -1,1

D = mask.*MM;
outside = (mask-1)*-1; % Inverse of mask
MMatrix = outside + D; % masking matrix M

MMatrix = diag(MMatrix);
end

