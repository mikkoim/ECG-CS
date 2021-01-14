clc; clear all;

PATH= 'F:\Dropbox\work\SAMI\compressive_sensing\orig_datasets\mitdb'; % location on MIT-BIH arrythmia dataset

SAMPLES2READ=512^2;         % number of samples to be read
d = dir(fullfile(PATH, '*.dat')); 

fields = {'folder', 'date', 'bytes', 'isdir', 'datenum'};
d = rmfield(d,fields);

for i = 1:numel(d)
    %------ SPECIFY DATA ------------------------------------------------------
    filename = d(i).name;
    [~,id,~] = fileparts(filename);
    
    
    HEADERFILE= [id '.hea'];      % header-file in text format
    ATRFILE= [id '.atr'];         % attributes-file in binary format
    DATAFILE= [id '.dat'];         % data-file

    [M, ANNOT, ATRTIMED, ANNOTD, TIME] = readMIT(PATH, HEADERFILE, ATRFILE, DATAFILE, SAMPLES2READ);
    d(i).id = id;
    d(i).M = M;
    d(i).ANNOT = ANNOT;
    d(i).ANNOTD = ANNOTD;
    d(i).TIME = TIME;
    d(i).ATRTIMED = ATRTIMED;
end

save('mitdb.mat','d');
