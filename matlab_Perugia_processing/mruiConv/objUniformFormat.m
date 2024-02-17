%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       GENERAL FORMAT FILE (AD)
%               v 0.2
%   2015-11-18: adaptions for multi dimensional mrui files
%   2015-05-08: this is the general format that needs to
%               be readed or writed by the plugins
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef objUniformFormat < handle
    properties
        header;
        metadata;
        
        predef;
        nospectra;
    end
    properties (Access = protected)
        data;        
    end
    methods
        function obj = objUniformFormat()
            obj.nospectra                 = false;
            obj.header.srcFilename        = '';
            obj.header.srcPath            = '';
            obj.header.srcFormat          = '';

            
            obj.header.dcminfo            = '';
            obj.header.nucleus            = -1;
            
            obj.header.date               = '';
            obj.header.patient.familyName = '';
            obj.header.dataPoints         = 0;
            obj.header.dataSets           = 0;
            obj.header.smpIntMs           = 0;
            obj.header.trnsFrequHz        = 0;
            obj.header.additionalInfo     = '';

            obj.metadata                  = '';
                      
            % 'real part FID' obj.data.values(1,:)            
            % 'imag part FID' obj.data.values(2,:)
            % 'real part Spec' obj.data.values(3,:)
            % 'imag part Spec' obj.data.values(4,:)
            obj.data.values     = zeros(4,0);
            
            % hard defined variable properties
            obj.predef.nuclei(:,1) = cellstr(char('H','P','C','F','Na'));
            obj.predef.nuclei(:,2) =             {0.0,1.0,2.0,3.0,4.0};            
        end
        
        function addFidData(obj, realPart, imagPart)
            % check if size of real and imag part are equal
            if ( size(realPart) ~= size(imagPart) )
                exceptionMsg = MException('DataError:sizeOfImagAndRealPart', 'The size of the imag and real part of the FID signal does not match.');        
                throw(exceptionMsg);                
            end
            % first idx = spectra; second idx = real/imag part; third idx =
            % points of FID
            obj.data.values = nan(size(realPart,1),4,size(realPart,2));
            
            obj.data.values(:,1,:) = realPart;
            obj.data.values(:,2,:) = imagPart;
            
            %obj.header.dataPoints = length(realPart);
        end
        
        function addSpecData(obj, realPart, imagPart)
            % check if size of real and imag part are equal
            if ( size(realPart) ~= size(imagPart) )
                exceptionMsg = MException('DataError:sizeOfImagAndRealPart', 'The size of the imag and real part of the Spec signal does not match.');        
                throw(exceptionMsg);                
            end            
            
            obj.data.values = nan(size(realPart,1),4,size(realPart,2));
            
            obj.data.values(3,:) = realPart;
            obj.data.values(4,:) = imagPart;
        end       
        
        function calcSpecData(obj)
            if (length(obj.data.values(1,:)) == 0 || length(obj.data.values(2,:)) == 0)
                exceptionMsg = MException('DataError:noDataAvailable', 'Please insert data for the FID signal.');        
                throw(exceptionMsg);                
            end
            spec = fftshift(fft(obj.data.values(1,:) + j*obj.data.values(2,:)));
            obj.data.values(3,:) = real(spec);
            obj.data.values(4,:) = imag(spec);
        end
        
        function [realFid, imagFid] = getFidData(obj)
            realFid = squeeze(obj.data.values(:,1,:));
            imagFid = squeeze(obj.data.values(:,2,:));
        end
        
        function [realSpec, imagSpec] = getSpecData(obj)
            % do calculation if still empty
            if ( isnan(obj.data.values(:,3,:)) )
                tmp = nan(size(obj.data.values,1),size(obj.data.values,3));
                for idx = 1:size(obj.data.values,1)
                    tmp(idx,:) = flip(fftshift(fft(obj.data.values(idx,1,:) + i*obj.data.values(idx,2,:))));
                end
            end
            realSpec = real(tmp(:,:));
            imagSpec = imag(tmp(:,:));            
        end
        
        function nucleusStr = getNucleusAsString(obj)
            try
                nucleusStr = obj.predef.nuclei{(find(cellfun(@(V) any(eq((V),obj.header.nucleus )), obj.predef.nuclei(:,2))==1)),1};
            catch
                nucleusStr = 'unknown';
            end
        end
        
        function setNucleusAsString(obj, nucleusStr)
            if (isempty((find(cellfun(@(V) any(strcmp((V),nucleusStr)), obj.predef.nuclei(:,1))==1))))
                obj.header.nucleus = -1.0;
            else
                obj.header.nucleus = obj.predef.nuclei{(find(cellfun(@(V) any(strcmp((V),nucleusStr)), obj.predef.nuclei(:,1))==1)),2};
            end
        end
    end
end