%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   2015-11-17 AD:  can be applied to sort the mrui files into specific
%                   directories concerning different criteria
%   2019-08-14 RR:  code revision and bug correction

%--------------------------------------------------------------------------
%                   Parameters:
%                       sortInfo.srcPath:       path of the mrui files
%                       sortInfo.seriesList:    list of series that should be considered
%                           files named as follow:
%                           xw_IMRM_yymmdd_[a]_[b]_[c]_[d]_[e].mrui
%                           [a] = 30xx or 70xx, 3=3T,7=7T, xx=session number
%                           [b] = n, patient reference (mostly always 1)
%                           [c] = sss, series number
%                           [d] = mmm, measurement number;
%                           [e] = aaa, averages number;
%                       sortInfo.dstPath:       path to store the sorted files
%                       sortInfo.type:          type of sorting that should be applied (i.e. 'dws')
%                       sortInfo.toFolders:     copy files of specific series into folders having unique identifying names
%                       sortInfo.toMrui:        generates a single mrui file considering all spectra from each series
%                       sortInfo.measType:      account hte typology of run sequence and the correspondence data acquired.
%                                               Especially here the focus is over the second dimension over in diffusion (dws) 
%                                               are varied gradients' strength or echo time (svs).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function sortMrui(sortInfo)
%     if (~isdeployed)
%         addpath('Internal Sources\mruiConv\');        
%     end    
    
    % series numbering from 1,2,3,... to 001,002,003,... according to data name
    sortInfo.seriesList = arrayfun(@(x) num2str(x,'%03.0f'), sortInfo.seriesList,'UniformOutput',false)';
    
    % cell array with a vector of series of interest named as follow:
    % xw_IMRM_yymmdd_30xx_1_ in other ords considering 1:22 char
    mruiList = cellfun(@(x) [char( unique( arrayfun(@(x) x.name(1:22), dir( [sortInfo.srcPath '\*.mrui'] ) , 'un', 0  ) ) ) '' x], sortInfo.seriesList, 'un', 0);
 
    % going to explore every series of interest
    waitbar(0, 'Please wait ...')
    for it=1:length(mruiList)
        
        waitbar(it/length(mruiList))
        % do sorting for dws spectral data as follow
        % FOLDER SORTING
        %   folders are named in the following way:
        %       b[xxxx]_m[yyy]_s[zzz], where xxxx is the b-values, yyy is
        %       the number of measurements and zzz is the series number
        % FILE SORTING
        %   further an mrui file is generated containing all spectra with
        %   filename:
        %        b[xxxx]_m[yyy]_s[zzz].mrui, where xxxx is the b-values, 
        %        yyy is the number of measurements and 
        %        number
        
        if strcmp(sortInfo.type,'dws')
            % retrive all mrui files with specific series nr
            seriesFiles = arrayfun(@(x) x.name, dir( [sortInfo.srcPath '\' mruiList{it} '*.mrui'] ) , 'un', 0  ); %* means copy all up to .mrui
            if isempty(seriesFiles)
                warning('No files found for series "%s.', sortInfo.seriesList{it});
            else
                % read information for sorting files
                [ ~, metaFile, ~ ] = cellfun(@(x) fileparts(x), seriesFiles,'UniformOutput',false);
                metaFile           = [metaFile{1} '.txt'];
                
                nrMeas = length(seriesFiles);
                if strcmp(sortInfo.measType,'dws')
                        bValue = 0;
                end
                if strcmp(sortInfo.measType,'svs')
                    bValue = str2double( getSpecMetaData( [sortInfo.srcPath '\' metaFile], 'alTE[0]' ) )/1000;
                end
                % sorting into folders
                if ( sortInfo.toFolders )                  
                    sortFolderStr = [sprintf( 's%s_m%03.0f',sortInfo.seriesList{it},nrMeas )]; %s = sortInfo..., %03.0f=nrMeas
                    % create folder
                    mkdir( sortInfo.dstPath, sortFolderStr );
                    % copy belonging files
                    copyfile( [sortInfo.srcPath '\' mruiList{it} '*'], [sortInfo.dstPath '\' sortFolderStr] ); 
                end
                % sorting into mrui files
                if ( sortInfo.toMrui ) 
                    %here is manipulated each .mrui file as binary file and merged in one .mrui file.
%                     [top_folder,fname] = fileparts(pwd);
%                     top_top_folder = fileparts(fileparts(pwd));
%                     addpath([ top_folder '\mruiConv\']);
%                     addpath([ top_top_folder 'External Sources\GetFullPath']);
                    addpath('External Sources\GetFullPath\');
                    addpath('Internal Sources\mruiConv\');
                    mruiReader = objMruiReader();
                    mruiReader.readFile( [sortInfo.srcPath '\' char(seriesFiles{1}) ] );
                    %mruiReader.readFile( [sortInfo.srcPath '\' builtin( '_brace', seriesFiles ) ] );
                    nrPts   = mruiReader.uniformFormat.header.dataPoints;
                    smpIntv = mruiReader.uniformFormat.header.smpIntMs;
                    trnsFre = mruiReader.uniformFormat.header.trnsFrequHz;
                    nucleus = mruiReader.uniformFormat.header.nucleus;                    
                    mruiSpecData = nan( nrMeas, nrPts );
                    for itMeas=1:nrMeas
                        mruiReader.readFile( [ sortInfo.srcPath '\' seriesFiles{itMeas} ] );
                        [realFid, imagFid]   = mruiReader.uniformFormat.getFidData();
                        mruiSpecData(itMeas,:) = flip(realFid)' + 1i*flip(imagFid)';
                    end

                    uniformFormat = objUniformFormat();
                    uniformFormat.header.dataPoints         = nrPts;
                    uniformFormat.header.smpIntMs           = smpIntv;
                    uniformFormat.header.trnsFrequHz        = trnsFre;
                    uniformFormat.header.nucleus            = nucleus;
                    uniformFormat.header.datasets           = nrMeas;
                    uniformFormat.header.patient.familyName = getSpecMetaData([sortInfo.srcPath '\' metaFile], 'PatientID');
                    uniformFormat.header.date               = getSpecMetaData([sortInfo.srcPath '\' metaFile], 'InstanceCreationDate');
                    uniformFormat.header.additionalInfo     = sprintf( 'b%05.2f_m%03.0f_s%s',round(bValue,2),nrMeas,sortInfo.seriesList{it} );
                    uniformFormat.addFidData(real(reshape(flip(mruiSpecData,2)', [nrMeas*nrPts 1])), imag(reshape(flip(mruiSpecData,2)', [nrMeas*nrPts 1])));
                                        
                    mruiWriter               = objMruiWriter();
                    mruiWriter.uniformFormat = uniformFormat;
                    sortFileStr = [sprintf( 's%s_m%03.0f',sortInfo.seriesList{it},nrMeas ) '.mrui'];
                    mruiWriter.writeFile([sortInfo.dstPath '\' sortFileStr]);
                end
            end
        end            
    end
    % remove all entries from list which do not have appropriate series nr
    disp('.mrui data sorting is done')
end

% helper function: it explores the .txt file looking for specific data
function data = getSpecMetaData(file, identifier)
    fid = fopen(file);
    FileInfo = dir(file);
    bufferSize = FileInfo.bytes;

    file_content = fread(fid,bufferSize,'uint8=>char')';
    %file content: char sheet given by the .txt file. It's keep in memory
    %the \n areas as Newline

    fclose(fid);
    
    value = textscan(file_content(findstr(file_content, identifier):end), '%s',1,'Delimiter','\n');
    %here value is a str in a cell array which withdraw the line in
    %file_content which start with identifier and end with Newline (e.g. 'alTE[0] = 35000')
    value = cell2mat(value{1});
    if ( findstr(value, '=') )
        data = value(findstr(value, '=')+2:end); %gettin' just the actual number
    elseif     ( findstr(value, '  ') ) %I may be looking for numbers separated by label by blank
        data = value(findstr(value, '  ')+2:end);     
    end
end