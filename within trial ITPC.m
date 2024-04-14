load('all_EEG_CON.mat')

for  subj = 1:length(all_EEG_CON)
    data = eeglab2fieldtrip(all_EEG_CON{subj,1}, 'preprocessing','none');
    
%     cfg = [];
%     cfg.toilim = [-0.25 0.05];
%     data_baseline = ft_redefinetrial(cfg, data);

    cfg = [];         
    cfg.method     = 'wavelet'; 
    cfg.keeptrials =  'yes';
    cfg.output     = 'fourier';	
    cfg.foi        = 1:1:20;
    % cfg.width      = 7;%ÕâÑùÉèÖÃµÄ»°£¬ËùÓÐÆµÂÊ»áÊÇÏàÍ¬µÄ 
    cfg.width      = linspace(3,7,length(cfg.foi)); %¸ù¾ÝÆµÂÊµ÷Õ?
    cfg.toi        = -0.248:0.004:-0.048;
    cfg.pad = 'nextpow2';% more efficient FFT computation than the default 'maxperlen'
    freq = ft_freqanalysis(cfg, data_baseline);
    
    itc = [];
    itc.label     = freq.label;
    itc.freq      = freq.freq;
    itc.time      = freq.time;
    itc.elec      = freq.elec;
    itc.dimord    = 'chan_freq_time';
    
    F = freq.fourierspctrm;   % copy the Fourier spectrum
    N = size(F,1);           % number of trials

    % compute inter-trial phase coherence (itpc)
    itc.itpc      = F./abs(F);         % divide by amplitude
    itc.itpc      = sum(itc.itpc,1);   % sum angles
    itc.itpc      = abs(itc.itpc)/N;   % take the absolute value and normalize
    itc.itpc      = squeeze(itc.itpc); % remove the first singleton dimension

    % compute inter-trial linear coherence (itlc)
    itc.itlc      = sum(F) ./ (sqrt(N*sum(abs(F).^2)));
    itc.itlc      = abs(itc.itlc);     % take the absolute value, i.e. ignore phase
    itc.itlc      = squeeze(itc.itlc); % remove the first singleton dimension
    
    ITC_con_baseline{1,subj} = itc;   
end

save ITC_con_theta_baseline.mat ITC_con_baseline

for  subj = 1:length(all_EEG_CON)
    data = eeglab2fieldtrip(all_EEG_CON{subj,1}, 'preprocessing','none');
    
    cfg = [];
   cfg.toilim = [0.0 0.2];
   data_activation = ft_redefinetrial(cfg, data);

    cfg = [];                
    cfg.method     = 'wavelet'; 
    cfg.keeptrials =  'yes';
    cfg.output     = 'fourier';	
    cfg.foi        = 1:1:20;
    % cfg.width      = 7;%ÕâÑùÉèÖÃµÄ»°£¬ËùÓÐÆµÂÊ»áÊÇÏàÍ¬µÄ 
    cfg.width      = linspace(3,7,length(cfg.foi)); %¸ù¾ÝÆµÂÊµ÷Õ?
    cfg.toi        = 0.0:0.004:0.2;
    cfg.pad = 'nextpow2';% more efficient FFT computation than the default 'maxperlen'
    freq = ft_freqanalysis(cfg, data_activation);
    
    itc = [];
    itc.label     = freq.label;
    itc.freq      = freq.freq;
    itc.time      = freq.time;
    itc.elec      = freq.elec;
    itc.dimord    = 'chan_freq_time';
    
    F = freq.fourierspctrm;   % copy the Fourier spectrum
    N = size(F,1);           % number of trials

    % compute inter-trial phase coherence (itpc)
    itc.itpc      = F./abs(F);         % divide by amplitude
    itc.itpc      = sum(itc.itpc,1);   % sum angles
    itc.itpc      = abs(itc.itpc)/N;   % take the absolute value and normalize
    itc.itpc      = squeeze(itc.itpc); % remove the first singleton dimension

    % compute inter-trial linear coherence (itlc)
    itc.itlc      = sum(F) ./ (sqrt(N*sum(abs(F).^2)));
    itc.itlc      = abs(itc.itlc);     % take the absolute value, i.e. ignore phase
    itc.itlc      = squeeze(itc.itlc); % remove the first singleton dimension
    
    ITC_con_activation{1,subj} = itc;   
end

save ITC_con_theta_activation.mat ITC_con_activation

%% average

cfg = [];   
cfg.keepindividual = 'no';
cfg.parameter = 'itpc';
grandavg_con_baseline = ft_freqgrandaverage(cfg, ITC_con_baseline{1,1}, ITC_con_baseline{1,2}, ITC_con_baseline{1,3}, ITC_con_baseline{1,4}, ...
            ITC_con_baseline{1,5}, ITC_con_baseline{1,6}, ITC_con_baseline{1,7}, ITC_con_baseline{1,8}, ITC_con_baseline{1,9}, ITC_con_baseline{1,10}, ...
            ITC_con_baseline{1,11}, ITC_con_baseline{1,12}, ITC_con_baseline{1,13}, ITC_con_baseline{1,14}, ITC_con_baseline{1,15}, ...
            ITC_con_baseline{1,16}, ITC_con_baseline{1,17}, ITC_con_baseline{1,18}, ITC_con_baseline{1,19}, ITC_con_baseline{1,20}, ...
            ITC_con_baseline{1,21}, ITC_con_baseline{1,22}, ITC_con_baseline{1,23}, ITC_con_baseline{1,24}, ITC_con_baseline{1,25}, ...
            ITC_con_baseline{1,26}, ITC_con_baseline{1,27}, ITC_con_baseline{1,28}, ITC_con_baseline{1,29}, ITC_con_baseline{1,30}, ...
            ITC_con_baseline{1,31}, ITC_con_baseline{1,32}, ITC_con_baseline{1,33});

grandavg_con_activation = ft_freqgrandaverage(cfg, ITC_con_activation{1,1}, ITC_con_activation{1,2}, ITC_con_activation{1,3}, ITC_con_activation{1,4}, ...
            ITC_con_activation{1,5}, ITC_con_activation{1,6}, ITC_con_activation{1,7}, ITC_con_activation{1,8}, ITC_con_activation{1,9}, ITC_con_activation{1,10}, ...
            ITC_con_activation{1,11}, ITC_con_activation{1,12}, ITC_con_activation{1,13}, ITC_con_activation{1,14}, ITC_con_activation{1,15}, ...
            ITC_con_activation{1,16}, ITC_con_activation{1,17}, ITC_con_activation{1,18}, ITC_con_activation{1,19}, ITC_con_activation{1,20}, ...
            ITC_con_activation{1,21}, ITC_con_activation{1,22}, ITC_con_activation{1,23}, ITC_con_activation{1,24}, ITC_con_activation{1,25}, ...
            ITC_con_activation{1,26}, ITC_con_activation{1,27}, ITC_con_activation{1,28}, ITC_con_activation{1,29}, ITC_con_activation{1,30}, ...
            ITC_con_activation{1,31}, ITC_con_activation{1,32}, ITC_con_activation{1,33});
       
        
C_vs_IC = grandavg_con_activation;

C_vs_IC.powspctrm = grandavg_con_activation.itpc - grandavg_con_baseline.itpc;

%% within trial cluster based permutation test


cfg_layout = [];
cfg_layout.layout = 'EEG29.lay';
cfg_neighb           = [];
cfg_neighb.layout = ft_prepare_layout(cfg_layout);

cfg.layout = cfg_neighb.layout ;
cfg_neighb.method = 'distance';
cfg_neighb.feedback = 'yes';
cfg_neighb.neighbourdist = 0.25;
cfg.neighbours       = ft_prepare_neighbours(cfg_neighb, grandavg_con_activation);

cfg = [];
cfg.latency          = [0.0 0.22];
cfg.frequency        = [5 10];
cfg.avgovertime = 'yes'; %»»³É'no'ÊÔÊÔ£¨Ã¿¸öÊ±¼äµã·Ö±ðÍ³¼Æ£©
cfg.avgoverfreq = 'yes';
cfg.method           = 'montecarlo';
cfg.statistic        = 'ft_statfun_depsamplesT'; 
cfg.correctm         = 'cluster';
cfg.clusteralpha     = 0.05;%³õÊ¼µÄãÐÏÞ
cfg.clusterstatistic = 'maxsum';
cfg.minnbchan        = 0;
cfg.clustertail      = 0;
cfg.tail             = 0;
cfg.alpha            = 0.025;%×ûòÕãÐÏÞ£¨Ë«Î²¼?ÑéÐèÒªÊÇ0.025£©
cfg.numrandomization = 2000;

cfg_layout = [];
cfg_layout.layout = 'EEG29.lay';
cfg_neighb           = [];
cfg_neighb.layout = ft_prepare_layout(cfg_layout);

cfg.layout = cfg_neighb.layout ;
cfg_neighb.method = 'distance';
cfg_neighb.feedback = 'yes';
cfg_neighb.neighbourdist = 0.25;
cfg.neighbours       = ft_prepare_neighbours(cfg_neighb, grandavg_activation);

subj = 33; %±»ÊÔÊ?Ä¿£¨×éÄÚ£©
design = zeros(2,2*subj);
for i = 1:subj
  design(1,i) = i;
end
for i = 1:subj
  design(1,subj+i) = i;
end
design(2,1:subj)        = 1;
design(2,subj+1:2*subj) = 2;

cfg.design   = design;
cfg.uvar     = 1; %±»ÊÔ±äÁ¿µÚÒ»ÐÐ
cfg.ivar     = 2; %×Ô±äÁ¿µÚ¶?ÐÐ

cfg.parameter = 'itpc';

[stat] = ft_freqstatistics(cfg, ITC_con_activation{:}, ITC_con_baseline{:});   
