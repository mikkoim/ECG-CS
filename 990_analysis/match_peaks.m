function [precision, recall] = match_peaks(R,RA)
D=[];
for i = 1:length(R)
    r = R(i);
    for ii = 1:length(RA)
        ra = RA(ii);
        D(i,ii) = abs(r-ra);
    end
end

mins_RA = min(D);
mins_R = min(D,[],2);

precision = sum(mins_RA < 10)/length(RA);
recall = sum(mins_R < 10)/length(R);

end