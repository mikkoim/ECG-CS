function [B, F] = make_B(params)
% Watermark measurement matrix B
T = params.M;
m = params.m;

cont = false;
    while ~cont
        try
            B = sqrt(1/m)*randn(m,T);

            F = null(B','r')'; % Annihilation matrix
            F = orth(F')';

            assert(sum(sum(abs(F*B))) < 1e-8, 'F does not annihilate B')
            cont = true;
        catch ME
            continue;
        end
    end
    
end

