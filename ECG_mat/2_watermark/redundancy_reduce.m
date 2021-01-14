function [w] = redundancy_reduce(w3)
n = length(w3);
w = reshape(w3, n/3, 3);
w = mode(w,2);
end

