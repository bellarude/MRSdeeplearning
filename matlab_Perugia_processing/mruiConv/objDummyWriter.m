%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       parent calls for writer plugin development (AD)
%               v 0.1
%   2015-05-08: all new writer plugins have to be derived
%               from the objDummyReader class
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef objDummyWriter < handle
    properties
        uniformFormat = objUniformFormat; % for definition of the objUniformFormat look at "objUniformFormat.m"           
    end    
    properties (Abstract)
        uniqueFormateToken;
    end
    methods (Abstract)
         writeFile(file);
    end
    methods
        function name = getClassName(obj)
            name = class(obj);
        end
        function name = getParentClassName(obj)
            name = 'objDummyWriter';
        end
    end
end