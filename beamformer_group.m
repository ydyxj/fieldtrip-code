clc;clear all;close all
ft_defaults 

%% analyse congruent condition
%%
%%
load('all_EEG_CON.mat')
for  subj = 1:length(all_EEG_CON)
    data = eeglab2fieldtrip(all_EEG_CON{subj,1}, 'preprocessing','none');
    
    cfg = [];
    cfg.channel = 'all';
    cfg.reref = 'yes';
    cfg.refmethod = 'avg';
    cfg.implicitref = 'POz' % the implicit (non-recorded) reference channel is added to the data representation
    cfg.refchannel = { 'F7' 'F3' 'Fz' 'F4'	'F8' 'FC5' 'FC1' 'FC2' 'FC6' 'T7' 'T8' 'C3' 'Cz' 'C4' 'CP5' 'CP1' 'CP2' 'CP6' 'P7' 'P3' 'Pz' 'P4' 'P8' 'O1' 'Oz' 'O2'};

    cfg.toilim = [-0.3 -0.1];                       
    dataPre = ft_redefinetrial(cfg, data);
   
    cfg.toilim = [0.1 0.3];                       
    dataPost = ft_redefinetrial(cfg, data);
    %
    cfg = [];
    cfg.method    = 'mtmfft';
    cfg.output    = 'powandcsd'; % pow and cross-spectral density
    cfg.tapsmofrq = 5;
    cfg.foilim    = [25 25];
    freqPre = ft_freqanalysis(cfg, dataPre);

    cfg = [];
    cfg.method    = 'mtmfft';
    cfg.output    = 'powandcsd';
    cfg.tapsmofrq = 5;
    cfg.foilim    = [25 25];
    freqPost = ft_freqanalysis(cfg, dataPost);
   
    %set head model
    templateheadmodel = 'D:\Study\fieldtrip-20220321\template\headmodel\standard_bem.mat';
    load(templateheadmodel); % vol

    cfg                 = [];
    cfg.elec =  freqPost.elec;     % sensor positions
    cfg.headmodel = vol;        % volume conduction model
    cfg.reducerank      = 3;
    cfg.channel         = 'all';
    cfg.grid.resolution = 1;   % use a 3-D grid with a 1 cm resolution
    cfg.grid.unit       = 'cm';
    leadfield = ft_prepare_leadfield(cfg);
   
    %
    dataAll = ft_appenddata([], dataPre, dataPost); % append data
    cfg = [];
    cfg.method    = 'mtmfft';
    cfg.output    = 'powandcsd'; 
    cfg.tapsmofrq = 5;
    cfg.foilim    = [25 25];
    freqAll = ft_freqanalysis(cfg, dataAll);

    %source analysis
    cfg              = [];
    cfg.method       = 'dics'; 
    cfg.frequency    = 25;
    cfg.grid         = leadfield;
    cfg.headmodel    = vol;
    cfg.dics.projectnoise = 'yes'; % estimate noise
    cfg.dics.lambda       = '5%'; % how to regularise
    cfg.dics.keepfilter   = 'yes';  % keep the spatial filter in the output
    cfg.dics.realfilter   = 'yes'; % retain the real values
    sourceAll = ft_sourceanalysis(cfg, freqAll);
    %
    cfg.grid.filter = sourceAll.avg.filter;
    sourcePre_inc  = ft_sourceanalysis(cfg, freqPre );
    sourcePost_con = ft_sourceanalysis(cfg, freqPost);

    sourceDiff = sourcePost_con;
    sourceDiff.avg.pow = (sourcePost_con.avg.pow - sourcePre_inc.avg.pow) ./ sourcePre_inc.avg.pow;
    
    sourceDiff_con{1,subj} = sourceDiff;
    
    waitbar(subj/length(all_EEG_CON))
end

save sourceDiff_con.mat sourceDiff_con


%% analyse congruent condition
load('all_EEG_INC.mat')
for  subj = 1:length(all_EEG_INC)
    data = eeglab2fieldtrip(all_EEG_INC{subj,1}, 'preprocessing','none');
    
    %
    cfg = [];
    cfg.channel = 'all';
    cfg.reref = 'yes'; 
    cfg.refmethod = 'avg';
    cfg.implicitref = 'POz' % the implicit (non-recorded) reference channel is added to the data representation
    cfg.refchannel = { 'F7' 'F3' 'Fz' 'F4'	'F8' 'FC5' 'FC1' 'FC2' 'FC6' 'T7' 'T8' 'C3' 'Cz' 'C4' 'CP5' 'CP1' 'CP2' 'CP6' 'P7' 'P3' 'Pz' 'P4' 'P8' 'O1' 'Oz' 'O2'};
                                    
    cfg.toilim = [-0.3 -0.1];                       
    dataPre = ft_redefinetrial(cfg, data);
   
    cfg.toilim = [0.1 0.3];                       
    dataPost = ft_redefinetrial(cfg, data);
    %
    cfg = [];
    cfg.method    = 'mtmfft';
    cfg.output    = 'powandcsd'; % pow and cross-spectral density
    cfg.tapsmofrq = 5;
    cfg.foilim    = [25 25];
    freqPre = ft_freqanalysis(cfg, dataPre);

    cfg = [];
    cfg.method    = 'mtmfft';
    cfg.output    = 'powandcsd';
    cfg.tapsmofrq = 5;
    cfg.foilim    = [25 25];
    freqPost = ft_freqanalysis(cfg, dataPost);
    %
    templateheadmodel = 'D:\Study\fieldtrip-20220321\template\headmodel\standard_bem.mat';
    load(templateheadmodel); % vol

    cfg                 = [];
    cfg.elec =  freqPost.elec;     % sensor positions
    cfg.headmodel = vol;        % volume conduction model
    cfg.reducerank      = 3;
    cfg.channel         = 'all';
    cfg.grid.resolution = 1;   % use a 3-D grid with a 1 cm resolution
    cfg.grid.unit       = 'cm';
    leadfield = ft_prepare_leadfield(cfg);
    %
    dataAll = ft_appenddata([], dataPre, dataPost); 
    cfg = [];
    cfg.method    = 'mtmfft';
    cfg.output    = 'powandcsd'; 
    cfg.tapsmofrq = 5;
    cfg.foilim    = [25 25];
    freqAll = ft_freqanalysis(cfg, dataAll);

    cfg              = [];
    cfg.method       = 'dics'; 
    cfg.frequency    = 25;
    cfg.grid         = leadfield;
    cfg.headmodel    = vol;
    cfg.dics.projectnoise = 'yes';
    cfg.dics.lambda       = '5%';
    cfg.dics.keepfilter   = 'yes';
    cfg.dics.realfilter   = 'yes';
    sourceAll = ft_sourceanalysis(cfg, freqAll);
    %
    cfg.grid.filter = sourceAll.avg.filter;
    sourcePre_inc  = ft_sourceanalysis(cfg, freqPre );
    sourcePost_con = ft_sourceanalysis(cfg, freqPost);

    sourceDiff = sourcePost_con;
    sourceDiff.avg.pow = (sourcePost_con.avg.pow - sourcePre_inc.avg.pow) ./ sourcePre_inc.avg.pow;
    
    sourceDiff_inc{1,subj} = sourceDiff;
    
    waitbar(subj/length(all_EEG_INC))
end

save sourceDiff_inc.mat sourceDiff_inc
