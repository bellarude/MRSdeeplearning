% Rudy 240217
% data read from python preadicted/denoised + savings on mrui format

clear all, close all, clc
folderPath = 'C:\Users\Rudy\Documents\WMD\PhD_Bern\01_Project 2 - Deep Learning\Data_Perugia_denosing\';
load([folderPath 'Perugia_DL_PRED.mat']) %denoised
load([folderPath 'Perugia_input.mat']) %noisy
p = ((pred(:,:,1) + j*pred(:,:,2))/100)';
i = ((test(:,:,1) + j*test(:,:,2))/100)';

d = p-i;

%%
figure, 
subplot(211)
plot(real(i(:,15))), hold on
plot(real(p(:,15)))
subplot(212)
plot(real(d(:,15)))
%% saving to .mrui
 uniformFormat = objUniformFormat();

uniformFormat.header.dataPoints         = 1024;        % number of points (usually the length of the vector) 
uniformFormat.header.smpIntMs           = 0.25;      % sampling interval in ms
uniformFormat.header.trnsFrequHz        = 123253125;    % transmitter frequency
uniformFormat.header.nucleus            = 0;            % 'H' = 0,'P' = 1,'C' = 2,'F' = 3,'Na' = 4
uniformFormat.header.datasets           = 25;        % used for multi-dim spectra
uniformFormat.header.patient.familyName = '';    % chose a name you prefer (not needed)
uniformFormat.header.date               = '';           % string with the measurement date (not needed)
uniformFormat.header.additionalInfo     = '';           % some additional comments (not needed)

p_save = ifft(fftshift(p));
uniformFormat.addFidData(real(p_save), imag(p_save));
% Write data of metabolite signal to mrui file
mruiWriter               = objMruiWriter();
mruiWriter.uniformFormat = uniformFormat;

name = ['Perugia_DL_PRED'];
mruiWriter.writeFile( [ folderPath name '.mrui'] );

i_save = ifft(fftshift(i));
uniformFormat.addFidData(real(i_save), imag(i_save));
% Write data of metabolite signal to mrui file
mruiWriter               = objMruiWriter();
mruiWriter.uniformFormat = uniformFormat;

name = ['Perugia_input'];
mruiWriter.writeFile( [ folderPath name '.mrui'] );

d_save = ifft(fftshift(d));
uniformFormat.addFidData(real(d_save), imag(d_save));
% Write data of metabolite signal to mrui file
mruiWriter               = objMruiWriter();
mruiWriter.uniformFormat = uniformFormat;

name = ['Perugia_error'];
mruiWriter.writeFile( [ folderPath name '.mrui'] );
