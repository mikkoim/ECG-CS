function [L] = process_label(ANNt)


        if ((ANNt == 1) || (ANNt == 2) || (ANNt == 3) || (ANNt == 34) || (ANNt == 11))
           % N type: Normal, LBBB, RBBB, atrial escape, nodal escape
            L = 1;
            typ= 'N';% type

        elseif ((ANNt == 9) || (ANNt == 8) || (ANNt == 4) || (ANNt == 7))

            % S type: abAtrial, SVC, Atrial, nodal premature 
            L = 2;
            typ= 'S';

        %scnt = scnt + 1;

        elseif ((ANNt == 5) || (ANNt == 10))

            % V type: PVC, venricular escape
            L = 3;
            typ= 'V';

        %vcnt = vcnt + 1;    

        elseif (ANNt == 6)

            % F type: fusion of ventricular and normal
            L = 4;
            typ= 'F';

        %fcnt = fcnt + 1;    

        elseif ((ANNt == 13)) %| (ANNt == 12) | (ANNt == 38))

            % Q type: unknown
            L = 5;
            typ= 'Q';

        %qcnt = qcnt + 1;    

        else
            L = 0;
            typ='Others';

        %continue;   % ignore others    

        end  


end