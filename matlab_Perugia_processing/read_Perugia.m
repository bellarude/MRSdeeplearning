% MRS - SV Philips
% Rudy 240217
% data read from Philips dicom
clear all, close all, clc
folderPath = 'C:\Users\Rudy\Documents\WMD\PhD_Bern\01_Project 2 - Deep Learning\Data_Perugia_denosing\';
folderName ='dicom\';
nSpectra = 25;

addpath('mruiConv\');
for i = 1:nSpectra
    name_i = [ num2str(i) '.dcm'];
    dinfo=dicominfo([folderPath folderName name_i]);

    a = dinfo.SpectroscopyData;

    k = 1;

    for n = 1:1:length(a)

        if mod(n,2)

            real_part(k) = a(n);

        else

            imag_part(k) = a(n);

            k = k + 1;

        end

    end

    % FID structure:

    % 1:4096 -> r1,i1,r2,i2, ...,r4096,i4096

    % 1:2048 metabolites (water suppressed)

    % 2049:4096 water

    s_fid = (real_part(1:1024)) + j*(imag_part(1:1024));
    w_fid = (real_part(1025:2048)) + j*(imag_part(1025:2048));

    %
    % pseudo eddy current correction
    % Rudy: can be done much better!!!
    %
    corr_s_fid=s_fid./exp(j*angle(w_fid));

    %savings
    uniformFormat = objUniformFormat();

    uniformFormat.header.dataPoints         = 1024;        % number of points (usually the length of the vector) 
    uniformFormat.header.smpIntMs           = 0.25;      % sampling interval in ms
    uniformFormat.header.trnsFrequHz        = 123253125;    % transmitter frequency
    uniformFormat.header.nucleus            = 0;            % 'H' = 0,'P' = 1,'C' = 2,'F' = 3,'Na' = 4
    uniformFormat.header.datasets           = 1;        % used for multi-dim spectra
    uniformFormat.header.patient.familyName = '';    % chose a name you prefer (not needed)
    uniformFormat.header.date               = '';           % string with the measurement date (not needed)
    uniformFormat.header.additionalInfo     = '';           % some additional comments (not needed)


    uniformFormat.addFidData(real(corr_s_fid), imag(corr_s_fid));
    % Write data of metabolite signal to mrui file
    mruiWriter               = objMruiWriter();
    mruiWriter.uniformFormat = uniformFormat;
    saveToFolder = 'mruiTEST\';
    name = ['s' num2str(i)];
    mruiWriter.writeFile( [folderPath saveToFolder name '.mrui'] );
end

%% unused!

% % Apodization:
% 
% % exp(-t/Tmax); t=n*2000/(length(a)/4) in ms per TR=2000
% 
% % per Tmax=800 ms il fattore diventa exp(-n*0.0024)
% 
% %
% 
% for n = 1:1:length(a)/4
% 
%     corr_s_fid(n)=corr_s_fid(n)*exp(-n*0.0024);
% 
% end
% 
% %
% 
% % zerofill
% 
% %
% 
% corr_s_fid=horzcat(corr_s_fid,zeros(1,length(a)/4));
% 
% %
% 
% % spettro
% 
% %
% 
% s_spectro=fft(s_fid);
% 
% w_spectro=fft(w_fid);
% 
% corr_s_spectro=(fft(corr_s_fid));
% 
% %
% 
% % Plotting ******
% 
% %
% 
% % zoom + index rotation for spectro
% 
% for i=1:1:300
% 
%     ps_spectro(i)=s_spectro(1000-i);
% 
% end
% 
% % zoom + index rotation for spectro corrected
% 
% for i=1:1:600
% 
%     pcorr_s_spectro(i)=corr_s_spectro(2000-i);
% 
% end
% 
% % index rotation for water unsuppressed spectro
% 
% for i=1:1:511
% 
%     pw_spectro(i)=w_spectro(512-i);
% 
%     rw_spectro(i+511)=w_spectro(1025-i);
% 
% end
% 
% % figure
% 
% % plot(real(rw_spectro))
% 
% figure('Name','Simple FFT')
% 
% plot(real(ps_spectro))
% 
% figure('Name','Apodizzation and zero fill')
% 
% plot(real(pcorr_s_spectro))
% 
% %%
% 
% % fitting
% 
% %
% 
% x = [1:600];
% 
% gaussEqn = 'a1*exp(-((x-a2)/a3)^2)+b1*exp(-((x-b2)/b3)^2)+c1*exp(-((x-c2)/c3)^2)+d1*exp(-((x-d2)/d3)^2)+e1*exp(-((x-e2)/e3)^2)+f1*exp(-((x-f2)/f3)^2)+g1*exp(-((x-g2)/g3)^2)+h1*exp(-((x-h2)/h3)^2)+i1*exp(-((x-i2)/i3)^2)+j1*exp(-((x-j2)/j3)^2)';
% 
% startPoints = [1 300 7 0.1 265 5 0.1 255 5 0.1 242 5 0.2 165 6 0.2 143 6 0.2 97 6 0.5 89 5 0.2 71 10 02 50 6];
% 
% f2 = fit(x',real(pcorr_s_spectro)',gaussEqn,'Start', startPoints, 'Exclude', x < 100, 'Exclude', x > 350)
% 
% figure('Name','fitting')
% 
% plot(f2,x,real(pcorr_s_spectro))
% 
% %
% 
% % export
% 
% %
% 
% % met_fid = real_part(1:1024) + j*imag_part(1:1024);
% 
% % save('met_fid','met_fid')
% 
% % save('met_spectro','corr_s_spectro')
% 
% % github:
% 
% % 202101231959 Aragog linux
% 
% plot(real(ps_spectro)-real(pcorr_s_spectro))
% 
% % 202101281536 aggancio dirac (linux)