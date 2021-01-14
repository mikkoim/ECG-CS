clear all
load mitdb_block.mat

rng(123);
iN = randsample(find(y==1),60);
iS = randsample(find(y==2),60);
iV = randsample(find(y==3),60);
iF = randsample(find(y==4),60);

sample_inds = sort(cat(1,iN,iS,iV,iF));

inv_sample_inds = transpose(1:length(y));

inv_sample_inds(sample_inds) = [];

inv_X_sample = X(inv_sample_inds,:);
inv_y_sample = y(inv_sample_inds,:);

info_inv_sample.inds = info.inds(inv_sample_inds);
info_inv_sample.annot = info.annot(inv_sample_inds);
info_inv_sample.labels = info.labels(inv_sample_inds);
info_inv_sample.id = info.id(inv_sample_inds);

X = inv_X_sample;
y = inv_y_sample;
info = info_inv_sample;
save('mitdb_inverse_sample', 'X','y','info');