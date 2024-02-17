% Rudy 240217
% data read from mrui to .mat

clear all, close all, clc
folderPath = 'C:\Users\Rudy\Documents\WMD\PhD_Bern\01_Project 2 - Deep Learning\Data_Perugia_denosing\spectra_for_1D_Unet\';
% folderName ='dicom\';

addpath('MRUI_processing\');
data = readMRUI();
%%
X_test_matlabPerugia = data.complexFid{1};

%%
%complex double 1024x25
save([folderPath 'dataPerugiaDL_0phCORR.mat'], 'X_test_matlabPerugia' );


