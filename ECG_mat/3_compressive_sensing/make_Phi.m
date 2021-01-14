function [Phi] = make_Phi(name, params)
N = params.N;

if strcmp(name,'dct')
    Wav = @(t) dct(t);
    
elseif strcmp(name, 'dct_mat')
    Phi = DCT_MAT(N);
    return 
else
    level = 5;
    Wav = @(t) wavedec(t,level,name);
end

I=eye(N);
for i=1:N
    PhiT(:,i)=Wav(I(:,i));
    %fprintf("%i/%i\n",i,N);
end

Phi=pinv(PhiT);

end

function [CN] = DCT_MAT(N)

CN = zeros(N);
for n=0:N-1
    for k=0:N-1
        if k==0
            CN(k+1,n+1)=sqrt(1/N);
        else
            CN(k+1,n+1)=sqrt(2/N)*cos(pi*(n+0.5)*k/N);
        end
    end
end

end

