%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       parent calls for reader plugin development (AD)
%               v 0.1
%   2015-05-08: all new reader plugins have to be derived
%               from the objDummyReader class
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef objDummyReader < handle
    properties
        uniformFormat = objUniformFormat;   % for definition of the objUniformFormat look at "objUniformFormat.m"
    end
    properties (Abstract)
        uniqueFormateToken;
    end
    methods (Abstract)
        readFile(file);
    end
    methods
        function name = getClassName(obj)
            name = class(obj);
        end
        function name = getParentClassName(obj)
            name = 'objDummyReader';
        end
    end
end