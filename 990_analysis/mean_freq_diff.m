function m = mean_freq_diff(S, SB)

N = size(S,1);
N_sample = size(S,2);

D = zeros(N, N_sample);
for i = 1:N
    St = dct(S(i,:));
    SBt = dct(SB(i,:));

    d = abs(St - SBt);
    D(i,:) = d;
end
    
m = mean(D,1);
end