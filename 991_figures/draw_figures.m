clear all;
folder = '..\99_masking\MIT_masked';

domain = 'frequency';
masktype = 'stat';
CR = '065';
user = 'A';

fname = [domain '_' masktype '_cr' CR '.mat'];

load(fullfile(folder,fname));

X(y==5,:) = [];
S(y==5,:) = [];
SA(y==5,:) = [];
SB(y==5,:) = [];
y(y==5,:) = [];


i = 46; % V beat: 10
s = S(i,:)';
[~, ~, ~, r, ~, ~]  = rpeakdetect(s,1,0.5,0);
sA = SA(i,:)';
[~, ~, ~, rA, ~, ~]  = rpeakdetect(sA,1,0.5,0);
sB = SB(i,:)';
[~, ~, ~, rB, ~, ~]  = rpeakdetect(sB,1,0.5,0);


figure();
verticalLines = @(x) arrayfun(@xline, x, 'uni', false);

subplot(311);
plot(s,'k-'); hold on;
plot(r,s(r),'kd');
%title('Original signal, class S');
axis([0,2048,0,1])
set(gca,'XTick',[])
set(gcf, 'Position',  [100, 100, 500, 150])

subplot(312);
plot(sA,'k-');hold on;
plot(rA,sA(rA),'ko');
%title('User A frequency masked signal, classified as V');
axis([0,2048,0,1])
set(gca,'XTick',[])
set(gcf, 'Position',  [100, 100, 500, 150])

subplot(313);
plot(sB,'k-');hold on;
plot(rB,sB(rB),'ko');
%title('User B fully recovered signal');
axis([0,2048,0,1])
set(gca,'XTick',[])
set(gcf, 'Position',  [100, 100, 500, 150])

y_w = debug.y_w;
plot(y_w,'k-');hold on;
%title('User B fully recovered signal');
axis([0,1331,-1,1])
set(gca,'XTick',[])
set(gcf, 'Position',  [100, 100, 500, 150])

