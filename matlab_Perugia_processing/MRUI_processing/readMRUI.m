function [ data ] = readMRUI()
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
    top_folder = fileparts(pwd);
    top_top_folder = fileparts(fileparts(pwd));
    addpath([ top_folder '\03_matlab_working_space\Internal Sources\mruiConv\']);
    addpath([ top_folder '\03_matlab_working_space\External Sources\GetFullPath']);

    mruiReader = objMruiReader();

    % input 
    [fileName,specPath] = uigetfile('c:\*.mrui','Select file to open','Multiselect','on');

    % data import
    disp('data importing: wait...')
    if iscell(fileName) %multiselect true
        wb = waitbar(0,'bigger import --> longer time!');
        names = cellfun(@(x) [specPath x], fileName,'un',0);
        data.signalNames = fileName;
        for i =1:size(names,2) 
            waitbar(i /size(names,2))
            mruiReader.readFile(names{i});
            [realFid_i,imagFid_i]=mruiReader.uniformFormat.getFidData();
            if size(realFid_i,2)>1
                realFid_i = realFid_i';
                imagFid_i = imagFid_i';
            end
            complexFid_i = realFid_i+1i*imagFid_i;
            data.realFid(i) = {realFid_i}; % cellarray helps dealing with single averages signals
            data.imagFid(i) = {imagFid_i};
            data.complexFid(i) = {complexFid_i};
            data.SmpInt(i) = mruiReader.uniformFormat.header.smpIntMs; % sampling interval [ms]
            data.Fs(i) = 1/(data.SmpInt(i)*1e-3); % sampling frequency [Hz]
            data.dataPoints(i) = mruiReader.uniformFormat.header.dataPoints;
            data.t(i) = {[0:data.SmpInt(i):data.dataPoints(i)*data.SmpInt(i)-data.SmpInt(i)]'};
            data.txFreq(i) = mruiReader.uniformFormat.header.trnsFrequHz/1e6; %[Mhz]
        end  
        delete(wb)  
    else % multiselect false
        names = [specPath fileName];
        data.signalNames = fileName;
        mruiReader.readFile(names);
        [realFid, imagFid] = mruiReader.uniformFormat.getFidData();
        if size(realFid,2)>1
                realFid = realFid';
                imagFid = imagFid';
        end
        complexFid = realFid+1i*imagFid;
        data.realFid = {realFid};
        data.imagFid = {imagFid};
        data.complexFid = {complexFid};
        data.SmpInt = mruiReader.uniformFormat.header.smpIntMs; % sampling interval [ms]
        data.Fs = 1/(data.SmpInt*1e-3); % sampling frequency [Hz]
        data.dataPoints = mruiReader.uniformFormat.header.dataPoints;
        data.t = {[0:data.SmpInt:data.dataPoints*data.SmpInt-data.SmpInt]'};
        data.txFreq = mruiReader.uniformFormat.header.trnsFrequHz/1e6; %[Mhz]
    end
   
    fprintf('data imported\n --> FT evaluation ongoing...\n');

    % dft
    data.ft = cellfun(@(x) fftshift(fft(x)), data.complexFid,'un',0); 
    data.ftR = cellfun(@(x) real(x), data.ft,'un',0);
    data.ftI = cellfun(@(x) imag(x), data.ft,'un',0);

    for i=1:size(data.dataPoints,2)
        data.freq{i} = [0:data.Fs(i)/data.dataPoints(i):data.Fs(i)-data.Fs(i)/data.dataPoints(i)]...
            -(data.Fs(i)-data.Fs(i)/data.dataPoints(i))/2 + 4.65*data.txFreq(i);
        % frequency evaluation consider relative values to the carrier
        % transmitter frequency (data.txFreq), which is at the water peak. It's
        % therefore also considered a shift according to typical water peak
        % position (4.65ppm)

        % NB: notation arrayfun(@(x)) can't be used if dimensions disagree

        data.ppm{i} = data.freq{i}/data.txFreq(i);
        % remember frequency values are already relative to the carrier freq.
    end
    disp('data are ready')
end

