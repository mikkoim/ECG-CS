function inds = findAfromB(a,b)

n = numel(a);

inds = zeros(n,1);

for i = 1:n
   inds(i) =  find( a(i) == b );
end

end