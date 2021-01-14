function [A] = make_A(params)
m = params.m;
N = params.N;

A = sqrt(1/m)*randn(m,N);
A = orth(A')';
end

