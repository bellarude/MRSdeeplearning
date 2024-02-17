% ConversionToMRUI
function toMRUI(real,imag,exp,path,numberOfSpectra,numberOfDatapoints)

top_folder = fileparts(pwd);
addpath([ top_folder '\mruiConv\']);

rePart = real;    % put in here re part of your spectra
imPart = imag;    % put in here im part of your spectra
exportTo = exp;

uniformFormat = objUniformFormat();

uniformFormat.header.dataPoints         = numberOfDatapoints;      % number of points (usually the length of the vector) 
uniformFormat.header.smpIntMs           = 0.25;      % sampling interval in ms
uniformFormat.header.trnsFrequHz        = 123201833; % transmitter frequency
uniformFormat.header.nucleus            = 0;        % 'H' = 0,'P' = 1,'C' = 2,'F' = 3,'Na' = 4
uniformFormat.header.datasets           = numberOfSpectra;        % used for multi-dim spectra
uniformFormat.header.patient.familyName = '???';    % chose a name you prefer (not needed)
uniformFormat.header.date               = '';       % string with the measurement date (not needed)
uniformFormat.header.additionalInfo     = '';       % some additional comments (not needed)
uniformFormat.addFidData(rePart, imPart);

% Write data of metabolite signal to mrui file
mruiWriter               = objMruiWriter();
mruiWriter.uniformFormat = uniformFormat;
mruiWriter.writeFile( path );