%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       read mrui files (AD)
%               v 0.2
%   2015-11-18: read in multi dimensional spectral mrui files
%   2015-05-08: allows to read in mrui files which are
%               stored in the transparent objUniformFormat
%               structure
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef objMruiReader < objDummyReader
    properties
        uniqueFormateToken = 'mrui';
    end
    methods
        function readFile(obj, file)
            % read header information
            fileHandle = fopen( file, 'rb'); %gotta be passed the whole path (from C:\Users\...) otherwise not working 

            headerVector    = double(nan(64,1));
            
            for it1=1:64
                headerVector(it1) = double(fread(fileHandle, 1, 'float64','s'));
            end
            
            [obj.uniformFormat.header.srcPath, ...
                obj.uniformFormat.header.srcFilename, ...
                obj.uniformFormat.header.srcFormat]    = fileparts(GetFullPath(file));
            obj.uniformFormat.header.dataPoints        = headerVector(2);
            obj.uniformFormat.header.smpIntMs          = headerVector(3);
            obj.uniformFormat.header.trnsFrequHz       = headerVector(6);
            obj.uniformFormat.header.nucleus           = headerVector(8);
            obj.uniformFormat.header.datasets          = headerVector(14);
            
            
            dataVector = double(nan(obj.uniformFormat.header.datasets,obj.uniformFormat.header.dataPoints,2));
            % read data from mrui format
            for it3=1:obj.uniformFormat.header.datasets
                for it2=1:obj.uniformFormat.header.dataPoints
                    dataVector(it3,it2,1) = double(fread(fileHandle, 1, 'float64','s'));    % real part
                    dataVector(it3,it2,2) = double(fread(fileHandle, 1, 'float64','s'));    % imag part
                end
            end
            
            obj.uniformFormat.addFidData(dataVector(:,:,1),dataVector(:,:,2));
            
            % read meta information
            meta = textscan(fileHandle,'%s','delimiter','\n');
            meta = meta{1,1};
            
            
            obj.uniformFormat.header.patient.familyName = meta{1,1};
            obj.uniformFormat.header.date               = meta{2,1};
            obj.uniformFormat.header.additionalInfo     = meta{4,1};
            
            fclose(fileHandle);
        end
    end
end