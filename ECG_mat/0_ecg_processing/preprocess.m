function [s] = preprocess(ecg, params)

s = ecg(1:params.N);
s = (s-min(s));
s = s./max(s);

s = reshape(s,params.N,1);
end

