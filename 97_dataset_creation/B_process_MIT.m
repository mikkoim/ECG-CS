load mitdb.mat %MIT database processed to fixed length
block_inds = 2048:2048:512^2; %Indices separating the blocks

dest_folder = 'F:\Dropbox\work\SAMI\compressive_sensing\CS\97_dataset_creation\MIT_block_ds';

for i = 1:numel(d)
    ecg = d(i).M(:,1); % The full signal
    annot_ecg = d(i).ANNOTD; % Full annotations
    annot_inds = findAfromB(d(i).ATRTIMED, d(i).TIME); % Annotation indices

    % Remove 'other' type labels and save only N,S,V,F,Q
    labels_ecg = arrayfun(@(x) process_label(x),annot_ecg); % Original annotation transformed to NSVFQ -labeling
    
    annot_inds(labels_ecg == 0) = []; % Remove 'other'
    annot_ecg(labels_ecg == 0) = [];
    labels_ecg(labels_ecg == 0) = []; 

    y = zeros(128,1); % Label of each shorter signal
    
    block_start = 1;
    for ii = 1:numel(block_inds)
        block_end = block_inds(ii);
        
        ind_inds = logical((annot_inds > block_start) .* (annot_inds <= block_end)); % Location vector of annotation positions
        
        inds_block = annot_inds(ind_inds) - block_start + 1;
        labels_block = labels_ecg(ind_inds);
        annot_block = annot_ecg(ind_inds);
        
        % Take only abnormal beats to account
        anomaly_labels = labels_block;
        anomaly_labels(labels_block == 1) = [];

        L = mode(anomaly_labels); % The final label of the block is the most common abnormal beat type
        
        if isnan(L)
            L = 1;
        end

        y(ii) = L;
        block_start = block_end;

        info.inds{ii,1} = inds_block;
        info.annot{ii,1} = annot_block;
        info.labels{ii,1} = labels_block;
    end
    
    % Process the signal to blocks
    ecg_block = reshape(ecg,[2048,128])';

    % Save to file
    fname = fullfile(dest_folder, [d(i).id '_block.mat']);
    save(fname, 'ecg_block', 'y', 'info');
    fprintf(sprintf("%i/%i\n",i,numel(d)));
end


%% Create a single data array

d = dir(fullfile(dest_folder,'*.mat'));

X = zeros(128*48,2048);
y_full = zeros(128*48,1);
info_f.inds = {};
info_f.annot = {};
info_f.labels = {};
info_f.id = {};

for i = 1:numel(d)
    load(fullfile(dest_folder,d(i).name));
    X((i-1)*128+1:(i)*128,:) = ecg_block;
    y_full((i-1)*128+1:(i)*128) = y;
    
    info_f.inds((i-1)*128+1:(i)*128,1) = info.inds;
    info_f.annot((i-1)*128+1:(i)*128,1) = info.annot;
    info_f.labels((i-1)*128+1:(i)*128,1) = info.labels;
    info_f.id((i-1)*128+1:(i)*128,1) = repmat({d(i).name(1:3)},128,1);
end

info = info_f;
y = y_full;
save('mitdb_block','X','y','info');


%% Sampling to create the masking dataset
rng(123);
iN = randsample(find(y==1),60);
iS = randsample(find(y==2),60);
iV = randsample(find(y==3),60);
iF = randsample(find(y==4),60);
iQ = find(y==5);

sample_inds = sort(cat(1,iN,iS,iV,iF, iQ));

X_sample = X(sample_inds,:);
y_sample = y(sample_inds,:);

info_sample.inds = info.inds(sample_inds);
info_sample.annot = info.annot(sample_inds);
info_sample.labels = info.labels(sample_inds);
info_sample.id = info.id(sample_inds);

X = X_sample;
y = y_sample;
info = info_sample;
save('mitdb_sample', 'X','y','info');

