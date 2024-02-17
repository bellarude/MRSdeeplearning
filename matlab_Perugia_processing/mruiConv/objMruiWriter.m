%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       writing of spectral information into single mrui files (AD)
%               v 0.2
%   2015-11-18: writing of multi dimensional mrui files
%   2015-05-08: this plugin allows the writing of spectral 
%               information into single mrui files
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef objMruiWriter < objDummyWriter
    properties
        uniqueFormateToken = 'mrui';
    end
    methods
        function writeFile(obj, file)
            if (exist(file,'file') == 2)
                exceptionMsg = MException('FileError:destinationFileDuplicate', 'The destination file under "%s" already exists.', file);        
                throw(exceptionMsg);  
            end
            if (obj.uniformFormat.nospectra)
                exceptionMsg = MException('FileError:noSpectralInformation', 'The source file does not contain any spectral information.');        
                throw(exceptionMsg);  
            end
            
            % define the byte position for header information (index of dataVector is the byte position [int64])
            % for detailed information refere to manual .../jMRUI/doc/help/help.htm#QUANTITATION%20in%20COMMAND%20MODE
            dataVector     = double(nan(64,1));
            dataVector(1)  = 0;                                      % 1 for an FID (The most common data); 2 for a simulated Spectrum; 3 for a simulated FID; 4 for a peak table
            dataVector(2)  = obj.uniformFormat.header.dataPoints;    % datapoints
            dataVector(3)  = obj.uniformFormat.header.smpIntMs;      % sampling interval in ms
            dataVector(6)  = obj.uniformFormat.header.trnsFrequHz;   % transmitter frequency in Hz
            dataVector(8)  = double(obj.uniformFormat.header.nucleus);
            dataVector(14) = obj.uniformFormat.header.datasets;
            
            % write header information
            fileHandle = fopen( file, 'wb');

            for it1=1:64
                if ~isnan( dataVector(it1) )
                    byteData = dataVector(it1);
                    fwrite(fileHandle, byteData, 'float64','s');
                else
                    fwrite(fileHandle, 0.0, 'float64','s');
                end
            end

            % write data to mrui file            
            [realFid, imagFid] = obj.uniformFormat.getFidData();
            
            for it2=1:obj.uniformFormat.header.dataPoints*dataVector(14);
                fwrite(fileHandle, realFid(it2), 'float64','s');
                fwrite(fileHandle, imagFid(it2), 'float64','s');
            end

            % write additional meta information
            fwrite(fileHandle, sprintf('%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n', ...
                obj.uniformFormat.header.patient.familyName, obj.uniformFormat.header.date, ...
                '', obj.uniformFormat.header.additionalInfo, '', '', '', '', '', ''), 'char','s');
            fclose(fileHandle);            
        end
    end
end