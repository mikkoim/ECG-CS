clear all;
folder = '..\99_masking\MIT_masked';

domain = 'frequency';
masktype = 'stat';
CR = '065';
user = 'B';

fname = [domain '_' masktype '_cr' CR '.mat'];

load(fullfile(folder,fname));
e = info.errors;
success_rate = sum(e==0)/length(e);

X(y==5,:) = [];
S(y==5,:) = [];
SA(y==5,:) = [];
SB(y==5,:) = [];
e(y==5,:) = [];
y(y==5,:) = [];

X(e == Inf,:) = [];
S(e == Inf,:) = [];
SA(e == Inf,:) = [];
SB(e == Inf,:) = [];
y(e == Inf,:) = [];
e(e == Inf,:) = [];

for si = 1:size(X,1)
    s = S(si,:)';
    
    if strcmp(user,'A')
        shat = SA(si,:)';
    elseif strcmp(user,'B')
        shat = SB(si,:)';
    end
    
    psnrs(si,1) = psnr(s,shat);
    
    [~, ~, ~, r, ~, ~]  = rpeakdetect(s,1,0.5,0);
    [~, ~, ~, ra, ~, ~]  = rpeakdetect(shat,1,0.5,0);

    [precision, recall] = match_peaks(r,ra);

    P(si,1) = precision;
    R(si,1) = recall;

end

P(isnan(P)) = 0;
R(isnan(R)) = 0;

fprintf(sprintf("\nSuccess: %f\nPrecision: %f\nRecall: %f\nPSNR: %f\n\n", ...
    success_rate, ...
    mean(P), ...
    mean(R), ...
    mean(psnrs)));

