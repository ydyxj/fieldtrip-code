%% epoch congruent condition

Subj= {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'};
    % '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40'};	

% Import Data
for i = 1:length(Subj)

    %set name
    setname = strcat('s',num2str(i),'JP_fil_resamp_ica_epoch_ar','.set');  %s1JP_fil_resamp_ica_epoch_ar.set

    % import .set data
    EEG = pop_loadset('filename',setname,'filepath','D:\Study\eeg data\本実験\Raw data\JP\rejdata');
      
%epoch congruent

    EEG = pop_epoch( EEG, { 'B1(S11)'}, [-0.6  0.8], 'newname', ' resampled epochs', 'epochinfo', 'yes');
    EEG = pop_rmbase(EEG, [-600 0]); % Remove baseline from -200 to 0 ms


%notch filter [35]
    
% Apply a 4th-order Butterworth notch filter between 49.1-50.2 Hz
%EEG = pop_basicfilter(EEG, 1:31, 'Boundary', 'boundary', 'Cutoff', [49.1 50.2], 'Design', 'butter', 'Filter', 'bandstop', 'Order', 4, 'RemoveDC', 'on');

 %save to disc
    EEG = pop_saveset( EEG, 'filename',strcat('s',num2str(i),'JP_con','.set'),'filepath','D:\Study\eeg data\本実験\Raw data\JP\TFpre\condition\cong');

end

%% extract the trials of conditions(con) &separate data structures

dataPath = 'D:\Study\eeg data\本実験\Raw data\JP\TFpre\condition\cong'; % It depends on where the
                                 % wikiDatasets folder is
                                 % stored
savePath = 'D:\Study\eeg data\本実験\Raw data\JP\TF\condition\cong'
save_path='D:\Study\eeg data\本実験\Raw data\JP\TF\condition\cong\pre\'

Subj= {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'};	 


for i = 1:length(Subj)
    fileName = strcat('s',num2str(i),'JP_con','.set');%s1EN_fil_resamp_ica_epoch_ar.set s1fil_resamp_ica_epoch_lowf_basel
 
    dataSet = fullfile(dataPath,fileName);
   % load(dataSet); % loading the variable 'data'
    
    cfg = [];
   
    cfg.dataset = dataSet;
    cfg.reref = 'yes'; 
    cfg.channel = 'all';
    cfg.implicitref = 'POz' % the implicit (non-recorded) reference channel is added to the data representation
    cfg.refchannel = { 'C3' 'Cz' 'C4' 'CP5' 'CP1' 'CP2' 'CP6' 'P7' 'P3' 'Pz' 'P4' 'P8' 'O1' 'Oz' 'O2'};
    
  %set trialfun for conditional trial definition

    %cfg.outputfile  = fullfile(savePath,strcat('s',num2str(i),'data_con.mat'));

    condA =  ft_preprocessing (cfg);
    condA1= ft_timelockanalysis([], condA)

    save([save_path,'s', num2str(i), 'JP_con.mat'], 'condA') 

end

%% epoch incongruent condition

Subj= {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'}
   % '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40'};	

% Import Data
for i = 1:length(Subj)

    %set name
    setname = strcat('s',num2str(i),'JP_fil_resamp_ica_epoch_ar','.set');  %s1JP_fil_resamp_ica_epoch_ar.set

    % import .set data
    EEG = pop_loadset('filename',setname,'filepath','D:\Study\eeg data\本実験\Raw data\JP\rejdata');
    
%epoch incongruent

    EEG = pop_epoch( EEG, { 'B3(S13)'}, [-0.6  0.8], 'newname', ' resampled epochs', 'epochinfo', 'yes');
   
    EEG = pop_rmbase(EEG, [-600 0]); % Remove baseline from -200 to 0 ms

%,'B2(S12)','B3(S13)'

 %add zero padding
    
%notch filter [35]
% Apply a 4th-order Butterworth notch filter between 49.1-50.2 Hz
%EEG = pop_basicfilter(EEG, 1:31, 'Boundary', 'boundary', 'Cutoff', [49.1 50.2], 'Design', 'butter', 'Filter', 'bandstop', 'Order', 4, 'RemoveDC', 'on');

 %save to disc
    EEG = pop_saveset( EEG, 'filename',strcat('s',num2str(i),'JP_incon','.set'),'filepath','D:\Study\eeg data\本実験\Raw data\JP\TFpre\condition\incon');

end



%% between trails in single subject

%% extract the trials of conditions(incon) &separate data structures

dataPath3 = 'D:\Study\eeg data\本実験\Raw data\JP\TFpre\condition\incon'; % It depends on where the
                                 % wikiDatasets folder is
                                 % stored
savePath3 = 'D:\Study\eeg data\本実験\Raw data\JP\TF\condition\incon'
save_path3 ='D:\Study\eeg data\本実験\Raw data\JP\TF\condition\incon\pre\'
Subj= {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'};	 


for i = 1:length(Subj)
    fileName1 = strcat('s',num2str(i),'JP_incon','.set');%s1EN_fil_resamp_ica_epoch_ar.set s1fil_resamp_ica_epoch_lowf_basel
 
    dataSet1 = fullfile(dataPath3,fileName1);
   % load(dataSet); % loading the variable 'data'
    
    cfg = [];
    
    cfg.dataset = dataSet1;
    cfg.reref = 'yes'; 
    cfg.channel = 'all';
    cfg.implicitref = 'POz' % the implicit (non-recorded) reference channel is added to the data representation
    cfg.refchannel = { 'C3' 'Cz' 'C4' 'CP5' 'CP1' 'CP2' 'CP6' 'P7' 'P3' 'Pz' 'P4' 'P8' 'O1' 'Oz' 'O2'};
    
    %set trialfun for conditional trial definition

   % cfg.outputfile  = fullfile(savePath,strcat('s',num2str(i),'data_incon.mat'));
    condB = ft_preprocessing(cfg); 
    condB1 = ft_timelockanalysis([], condB);
    save([save_path3,'s', num2str(i), 'JP_incon.mat'], 'condB') 
    
end

%% wavelet & baseline correction for incongurent 

Subj= {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'};	 
dataPath1 = 'D:\Study\eeg data\本実験\Raw data\JP\TF\condition\incon\pre'; % It depends on where the
savedata1 = 'D:\Study\eeg data\本実験\Raw data\JP\TF\condition\incon\wavelet'; % It depends on where the
savedata11 = 'D:\Study\eeg data\本実験\Raw data\JP\TF\condition\incon\ar'; % It depends on where the

for i = 1:length(Subj)

    fileName2 = strcat('s',num2str(i),'JP_incon.mat');

    dataSet11 = fullfile(dataPath1,fileName2)


    cfg              = [];
    cfg.dataset      = dataSet11;
    data_eeg = ft_preprocessing(cfg)


    cfg.output       = 'pow'; %return the power-spectra
    cfg.channel      = 'all'; % default = 'all';
            
    cfg.pad          = 'nextpow2';% number, 'nextpow2', or 'maxperlen'
    
    cfg.keeptrials   = 'yes'; %necessary for subsequent statistical analysis
    cfg.foi          = 8:2:40;   % analysis 12 to 40 Hz in steps of 2 Hz
    cfg.method       = 'mtmconvol';  %'sum', 'svd', 'abssvd', or 'complex' 
                                   % (default = 'sum')
    cfg.taper        = 'hanning'; %dpss for multitaper

    cfg.t_ftimwin    = 0.08./cfg.foi; 

    %0.03*ones(length(cfg.foi),1); %winsize;  or 7. /cfg.foi;  % 7 cycles per time window

    cfg.toi          = -0.6:0.01:0.6;
    cfg.demean       =   'yes'; %whether to apply baseline correction,  
                               % 'yes' or 'no' (default = 'no')
  
   
    TFRwave_Incon = ft_freqanalysis(cfg, data_eeg);

    cfg.baselinetype = 'db'; % 'absolute','relative','relchange' or 'db','zscore'
    cfg.baselinewindow = [-0.3 -0.1];
    
    
    cfg.outputfile  = fullfile(savedata1,strcat('s',num2str(i),'TFRwave_Incon.mat'));
    
    bslnTimeFreqAvg_Incon{i} = ft_freqbaseline(cfg,TFRwave_Incon);

   % Computing the average over trials
    cfg = [];
    cfg.outputfile  = fullfile(savedata11,strcat('s',num2str(i),'TFRwave_Incon_ar.mat'));
    bslnTimeFreqAvg_Incon{i} = ft_freqdescriptives(cfg,bslnTimeFreqAvg_Incon{i});
    
end 

%% wavelet & baseline correction for congurent 
Subj= {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'};	 
dataPath2 = 'D:\Study\eeg data\本実験\Raw data\JP\TF\condition\cong\pre'; % It depends on where the
savedata2 = 'D:\Study\eeg data\本実験\Raw data\JP\TF\condition\cong\wavelet'; % It depends on where the
savedata22 = 'D:\Study\eeg data\本実験\Raw data\JP\TF\condition\cong\ar'; % It depends on where the


for i = 1:length(Subj)
    fileName3 = strcat('s',num2str(i),'JP_con','.mat');
    dataSet22 = fullfile(dataPath2,fileName3);
    %load(dataSet22); % loading the variable 'data'

    cfg              = [];
    cfg.dataset      = dataSet22;
    data_eeg = ft_preprocessing(cfg)


    cfg              = [];
    cfg.output       = 'pow'; %return the power-spectra
    cfg.channel      = 'all'; % default = 'all';
        
    cfg.pad          = 'nextpow2';% number, 'nextpow2', or 'maxperlen'
    
    cfg.keeptrials   = 'yes'; %necessary for subsequent statistical analysis
    cfg.foi          = 8:2:40;   % analysis 12 to 40 Hz in steps of 2 Hz
    cfg.method       = 'mtmconvol';  %'sum', 'svd', 'abssvd', or 'complex' 
                                   % (default = 'sum')
    cfg.taper        = 'hanning'; %dpss for multitaper

    cfg.t_ftimwin    = 0.08./cfg.foi; 

    %0.03*ones(length(cfg.foi),1); %winsize;  or 7. /cfg.foi;  % 7 cycles per time window

    cfg.toi          = -0.6:0.01:0.6;
    cfg.demean       =   'yes'; %whether to apply baseline correction,  
                               % 'yes' or 'no' (default = 'no')
                               
    TFRwave_Con = ft_freqanalysis(cfg,data_eeg);

    cfg.baselinetype = 'db'; % 'absolute','relative','relchange' or 'db'
    cfg.baselinewindow = [-0.3 -0.1];

    cfg.outputfile  = fullfile(savedata2,strcat('s',num2str(i),'TFRwave_Con.mat'));
    
    bslnTimeFreqAvg_Con{i} = ft_freqbaseline(cfg,TFRwave_Con);

   % Computing the average over trials
    cfg = [];
    cfg.outputfile  = fullfile(savedata22,strcat('s',num2str(i),'TFRwave_Con_ar.mat'));
    bslnTimeFreqAvg_Con{i} = ft_freqdescriptives(cfg,bslnTimeFreqAvg_Con{i});

end   

%% Grand average time-frequency data

savedata22 = 'D:\Study\eeg data\本実験\Raw data\JP\TF\condition\cong\ar'; % It depends on where the

for i = 1:length(Subj)
    fileName = strcat('s',num2str(i),'TFRwave_Con_ar.mat');
    dataSet1 = fullfile(savedata22,fileName);
    load(dataSet1); % loading the variable 'data'
    cfg = [];
    cfg.dataset      = dataSet1;
    
    gAvgBslnTimeFreq_Con = ft_freqgrandaverage(cfg,bslnTimeFreqAvg_Con{:});

end

% Incongruent
Subj= {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'};	 

savedata11 = 'D:\Study\eeg data\本実験\Raw data\JP\TF\condition\incon\ar'; % It depends on where the
for i = 1:length(Subj)
    fileName = strcat('s',num2str(i),'TFRwave_Incon_ar.mat');
    dataSet2 = fullfile(savedata11,fileName);
    load(dataSet2); % loading the variable 'data'
    cfg = [];
    cfg.dataset      = dataSet2;

    cfg = [];
    gAvgBslnTimeFreq_Incon = ft_freqgrandaverage(cfg,bslnTimeFreqAvg_Incon{:});


    C_vs_IC = gAvgBslnTimeFreq_Con;

    C_vs_IC.powspctrm = gAvgBslnTimeFreq_Con.powspctrm - gAvgBslnTimeFreq_Incon.powspctrm;

end


% Multi plot
cfg = [];
cfg.layout = 'easycapM1.mat';
cfg.interactive = 'yes';
cfg.showoutline = 'yes';
cfg.showlabels = 'yes';
cfg.fontsize = 10;
cfg.comment = sprintf('\n');
cfg.xlim = [0 0.6];
cfg.ylim = [13 40]; % we cannot analyze delta and theta band because 
                   % the trial lenght is too short
cfg.zlim = [-100 100]; 
cfg.colorbar = 'yes'; % or 'southoutside'
cfg.colormap = jet; % 'parula' or 'jet'

figure
ft_multiplotTFR(cfg,gAvgBslnTimeFreq_Con);
title('Grand average - Congruent (db)');

figure
ft_multiplotTFR(cfg,gAvgBslnTimeFreq_Incon);
title('Grand average - Incongruent (db)');

figure
ft_multiplotTFR(cfg,C_vs_IC);
title('Congruent (db) - Incongruent (db)');
 
% Plotting single average across electrodes of interest
%auditory cortex
cfg = [];
cfg.ylim = [13 40];
cfg.xlim = [0 0.6];
cfg.zlim = [-100 100];
cfg.channel = {'C3','Cz','C4'};
cfg.colormap = jet;
figure;
ft_singleplotTFR(cfg,gAvgBslnTimeFreq_Con);
set(gca,'Fontsize',20);
title('Auditory-Congruent');
set(gca,'box','on');
xlabel('time (s)');
ylabel('frequency (Hz)');
c = colorbar;
c.LineWidth = 1;
c.FontSize  = 18;
title(c,'db. change');

hold on
cfg = [];
cfg.ylim = [13 40];
cfg.xlim = [0 0.6];
cfg.zlim = [-100 100];
cfg.channel = {'C3','Cz','C4'};
cfg.colormap = jet;
figure;
ft_singleplotTFR(cfg,gAvgBslnTimeFreq_Incon);
set(gca,'Fontsize',20);
title('Auditory-Incongruent');
set(gca,'box','on');
xlabel('time (s)');
ylabel('frequency (Hz)');
c = colorbar;
c.LineWidth = 1;
c.FontSize  = 18;
title(c,'db. change');


cfg = [];
cfg.ylim = [13 40];
cfg.xlim = [0 0.6];
cfg.zlim = [-50 50];
cfg.channel = {'C3','Cz','C4'};
cfg.colormap = jet;
figure;
ft_singleplotTFR(cfg,C_vs_IC);
set(gca,'Fontsize',20);
title('Auditory-difference');
set(gca,'box','on');
xlabel('time (s)');
ylabel('frequency (Hz)');
c = colorbar;
c.LineWidth = 1;
c.FontSize  = 18;
title(c,'db. % change');

%visual cortex
cfg = [];
cfg.ylim = [8 50];
cfg.xlim = [0 0.6];
cfg.zlim = [-100 100];
cfg.channel = {'O1','Oz','O2'};
cfg.colormap = jet;
figure;
ft_singleplotTFR(cfg,gAvgBslnTimeFreq_Con);
set(gca,'Fontsize',20);
title('Visual-Congruent');
set(gca,'box','on');
xlabel('time (s)');
ylabel('frequency (Hz)');
c = colorbar;
c.LineWidth = 1;
c.FontSize  = 18;
title(c,'db. change');

hold on
cfg = [];
cfg.ylim = [8 50];
cfg.xlim = [0 0.6];
cfg.zlim = [-100 100];
cfg.channel = {'O1','Oz','O2'};
cfg.colormap = jet;
figure;
ft_singleplotTFR(cfg,gAvgBslnTimeFreq_Incon);
set(gca,'Fontsize',20);
title('Visual-Incongruent');
set(gca,'box','on');
xlabel('time (s)');
ylabel('frequency (Hz)');
c = colorbar;
c.LineWidth = 1;
c.FontSize  = 18;
title(c,'db. change');


cfg = [];
cfg.ylim = [8 50];
cfg.xlim = [0 0.6];
cfg.zlim = [-50 50];
cfg.channel = {'O1','Oz','O2'};
cfg.colormap = jet;
figure;
ft_singleplotTFR(cfg,C_vs_IC);
set(gca,'Fontsize',20);
title('Visual-difference');
set(gca,'box','on');
xlabel('time (s)');
ylabel('frequency (Hz)');
c = colorbar;
c.LineWidth = 1;
c.FontSize  = 18;
title(c,'db. % change');

%contourf

figure('color','w')
subplot(311);
contourf(gAvgBslnTimeFreq_Con.time,gAvgBslnTimeFreq_Con.freq,squeeze(mean(gAvgBslnTimeFreq_Con.powspctrm([16;17;18],:,:),1)),40,'linecolor','none')
set(gca,'xlim',[0 0.6], 'clim',[10 Inf],'ylim',[13 40])
title('Congruent','fontsize',20)
xlabel('Time(s)','fontsize',15)
ylabel('Frequency(Hz)','fontsize',15)
colorbar
h=colorbar;
set(get(h,'Title'),'string','Power (\mu V^2)','fontsize',15);
colormap(jet) 


subplot(312);contourf(gAvgBslnTimeFreq_Incon.time,gAvgBslnTimeFreq_Incon.freq,squeeze(mean(gAvgBslnTimeFreq_Incon.powspctrm([16;17;18],:,:),1)),40,'linecolor','none')
set(gca,'xlim',[0 0.6], 'clim',[10 Inf],'ylim',[13 40])
title('Incongruent','fontsize',20)
xlabel('Time(s)','fontsize',15)
ylabel('Frequency(Hz)','fontsize',15)
colorbar
h=colorbar;
set(get(h,'Title'),'string','Power (\mu V^2)','fontsize',15);
colormap(jet) 


subplot(313);contourf(gAvgBslnTimeFreq_Incon.time,gAvgBslnTimeFreq_Incon.freq,squeeze(mean(gAvgBslnTimeFreq_Con.powspctrm([16;17;18],:,:),1))-squeeze(mean(gAvgBslnTimeFreq_Incon.powspctrm([16;17;18],:,:),1)),40,'linecolor','none')
set(gca,'xlim',[0 0.6], 'clim',[-50 50],'ylim',[13 40])
title('auditory cortex difference','fontsize',20)
xlabel('Time(s)','fontsize',15)
ylabel('Frequency(Hz)','fontsize',15)
colorbar
h=colorbar;
set(get(h,'Title'),'string','Power (\mu V^2)','fontsize',15);
colormap(jet) 


foi = 13:30;
freq_idx = find(gAvgBslnTimeFreq_Con.freq >= foi(1) & gAvgBslnTimeFreq_Con.freq <= foi(end));
power1 = squeeze(mean(mean(gAvgBslnTimeFreq_Con.powspctrm([16;17;18],freq_idx,:),1),2)),
power2 = squeeze(mean(mean(gAvgBslnTimeFreq_Incon.powspctrm([16;17;18],freq_idx,:),1),2)),
%squeeze(mean(gAvgBslnTimeFreq_Con.powspctrm(:,freq_idx,:),3));
% plot time course of power
figure('color','w')
set(gca,'xlim',[0 0.6],'ylim',[0 50])
plot(gAvgBslnTimeFreq_Con.time, power1,'b','LineWidth',1.5);
hold on; plot(gAvgBslnTimeFreq_Con.time,power2,'r','LineWidth',1.5);
set(h,'Box','off','Fontsize',14, 'fontName', 'Arial')
xlabel('Time (s)');
ylabel('Power');
title('Time course of 13-30Hz power');



figure('color','w')
subplot(311);
contourf(gAvgBslnTimeFreq_Con.time,gAvgBslnTimeFreq_Con.freq,squeeze(mean(gAvgBslnTimeFreq_Con.powspctrm([29;30;31],:,:),1)),40,'linecolor','none')
set(gca,'xlim',[0 0.6], 'clim',[10 Inf],'ylim',[13 40])
title('Congruent','fontsize',20)
xlabel('Time(s)','fontsize',15)
ylabel('Frequency(Hz)','fontsize',15)
colorbar
h=colorbar;
set(get(h,'Title'),'string','Power (\mu V^2)','fontsize',15);
colormap(jet) 


subplot(312);contourf(gAvgBslnTimeFreq_Incon.time,gAvgBslnTimeFreq_Incon.freq,squeeze(mean(gAvgBslnTimeFreq_Incon.powspctrm([29;30;31],:,:),1)),40,'linecolor','none')
set(gca,'xlim',[0 0.6], 'clim',[10 Inf],'ylim',[13 40])
title('Incongruent','fontsize',20)
xlabel('Time(s)','fontsize',15)
ylabel('Frequency(Hz)','fontsize',15)
colorbar
h=colorbar;
set(get(h,'Title'),'string','Power (\mu V^2)','fontsize',15);
colormap(jet) 


subplot(313);contourf(gAvgBslnTimeFreq_Incon.time,gAvgBslnTimeFreq_Incon.freq,squeeze(mean(gAvgBslnTimeFreq_Con.powspctrm([29;30;31],:,:),1))-squeeze(mean(gAvgBslnTimeFreq_Incon.powspctrm([29;30;31],:,:),1)),40,'linecolor','none')
set(gca,'xlim',[0 0.6], 'clim',[-50 50],'ylim',[13 40])
title('auditory cortex difference','fontsize',20)
xlabel('Time(s)','fontsize',15)
ylabel('Frequency(Hz)','fontsize',15)
colorbar
h=colorbar;
set(get(h,'Title'),'string','Power (\mu V^2)','fontsize',15);
colormap(jet) 


foi = 13:30;
freq_idx = find(gAvgBslnTimeFreq_Con.freq >= foi(1) & gAvgBslnTimeFreq_Con.freq <= foi(end));
power1 = squeeze(mean(mean(gAvgBslnTimeFreq_Con.powspctrm([29;30;31],freq_idx,:),1),2)),
power2 = squeeze(mean(mean(gAvgBslnTimeFreq_Incon.powspctrm([29;30;31],freq_idx,:),1),2)),
%squeeze(mean(gAvgBslnTimeFreq_Con.powspctrm(:,freq_idx,:),3));
% plot time course of power
figure('color','w')
set(gca,'xlim',[0 0.6],'ylim',[0 50])
plot(gAvgBslnTimeFreq_Con.time, power1,'b','LineWidth',1.5);
hold on; plot(gAvgBslnTimeFreq_Con.time,power2,'r','LineWidth',1.5);
h=legend('Congruent','Incongruent') %% plot the average waveform for incongruent
set(h,'Box','off','Fontsize',14, 'fontName', 'Arial')
xlabel('Time (s)');
ylabel('Power');
title('Time course of 13-30Hz power');


figure;
plot(gAvgBslnTimeFreq_Con.time, squeeze(mean(mean(gAvgBslnTimeFreq_Con.powspctrm([16;17;18],:,:),1),3)), 'linewidth', 2)
hold on
plot(gAvgBslnTimeFreq_Incon.time, gAvgBslnTimeFreq_Incon.powspctrm(:), 'linewidth', 2)
legend('Congruent', 'Incongruent')
xlabel('Time(ms)')
ylabel('Power (\mu V^2)')

%% Plotting time-frequency analysis
% Defining channel layout
templateLayout = 'easycapM1.mat'; % one of the template layouts included
                                % in FieldTrip
cfg = [];
cfg.layout = which(templateLayout); 
layout = ft_prepare_layout(cfg);

% Increasing layout width and height
%if strcmp(templateLayout,'easycapM1.mat')
 %   layout.width = 0.07 * ones(length(layout.width),1);
  %  layout.height = 0.04 * ones(length(layout.height),1);
%end
 
cfg = [];
cfg.parameter = 'powspctrm';
cfg.channel      = 'all';
%cfg.baseline     = [-0.3 -0.1];
%cfg.baselinetype = 'absolute';
cfg.xlim         = [0.05 0.1];
cfg.zlim         = [10 Inf];
cfg.ylim         = [13 30];
cfg.marker       = 'on';
layout = 'easycapM1.mat'; 
cfg.layout       = layout;
figure('color','w')
ft_topoplotTFR(cfg, gAvgBslnTimeFreq_Con); title('Congruent stim');
colorbar
ft_topoplotTFR(cfg, gAvgBslnTimeFreq_Incon); title('Incongruent stim');
colorbar
ft_topoplotTFR(cfg,C_vs_IC); title('Difference');
colorbar

%% Statistical analysis of time-frequency representations at the group level
% Creating a neighbourhood structure to define how spatial clusters are formed

%% Compute the neighbours
templateElec = 'easycap-M1.txt';
elec = ft_read_sens(which(templateElec));
 
idx = ismember(elec.label,gAvgBslnTimeFreq_Con.label);
elec.chanpos = elec.chanpos(idx,:);
elec.chantype = elec.chantype(idx);
elec.chanunit = elec.chanunit(idx);
elec.elecpos = elec.elecpos(idx,:);
elec.label = elec.label(idx);


cfg = [];
cfg.method = 'distance'; %'template' or 'distance' or 'triangulation'
cfg.neighbourdist = 85; % maximum distance between neighbouring sensors
                        % (only for 'distance')
cfg.feedback = 'yes'; % show a neighbour plot
%cfg.grad      = grad; for MEG
neighbours = ft_prepare_neighbours(cfg,elec);


% Preparing the design matrix for the statistical evaluation
% For within-subjects analysis, the design matrix contains two rows
Subj= {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'};	 

Ns = length(Subj);

%cfg.design    = zeros(1, size(allsub,2));
design = [1:Ns 1:Ns; ones(1,Ns) ones(1,Ns)*2]; 
 
 
% Test the null-hypothesis of exchangeability between conditions
cfg = [];
cfg.channel = {'all'};
cfg.latency = 'all';
cfg.avgoverfreq = 'no';
cfg.method = 'montecarlo';
cfg.clusterthreshold = 'nonparametric_common';
cfg.correctm = 'cluster';
cfg.correcttail = 'prob';
cfg.clusterstatistic = 'maxsum';
cfg.minnbchan = 2;
cfg.neighbours = neighbours;
cfg.tail = 0;
cfg.clustertail = 0;
cfg.alpha = 0.025;
cfg.numrandomization = 1000;
cfg.statistic = 'ft_statfun_depsamplesT';
cfg.design = design;
cfg.uvar = 1; % row in design indicating subjects, repeated measure
cfg.ivar = 2; % row in design indicating condition for contrast

cfg.clusteralpha     = 0.01;
stat01 = ft_freqstatistics(cfg,bslnTimeFreqAvg_Con{:},bslnTimeFreqAvg_Incon{:});

cfg.clusteralpha     = 0.05;
stat05 = ft_freqstatistics(cfg,bslnTimeFreqAvg_Con{:},bslnTimeFreqAvg_Incon{:});

cfg.clusteralpha     = 0.05;
[stat] = ft_freqstatistics(cfg,bslnTimeFreqAvg_Con{:},bslnTimeFreqAvg_Incon{:});

stat.raweffect = gAvgBslnTimeFreq_Con.powspctrm - gAvgBslnTimeFreq_Incon.powspctrm;

save stat

%The format of the output
disp(stat)

% Plotting stat output over fronto-central electrodes
layout = 'D:\Study\eeg data\time-frequency analysis\easycapM1.mat';
cfg = [];
cfg.layout = layout;
cfg.channel = {'C3','Cz','C4'};
cfg.parameter = 'stat';
cfg.colormap = jet;
cfg.xlim = [0 0.6];
cfg.ylim = [13 40];
cfg.renderer  = 'painters';
cfg.marker = 'on';
cfg.style = 'fill';
cfg.comment = 'off';
cfg.maskparameter = 'mask';
cfg.maskstyle = 'outline';
cfg.colorbar = 'yes';
cfg.zlim = [-2.5 2.5];
figure;
ft_singleplotTFR(cfg,stat05);
set(gca, 'Fontsize',20);
title ('Mean over auditory-t-score (not corrected)');
set(gca,'box','on');
xlabel('time (s)');
ylabel('frequency (Hz)');
c = colorbar;
c.LineWidth = 1;
c.FontSize  = 18;
title(c,'t-val');

% Plotting stat output over occipital electrodes

cfg = [];
cfg.layout = layout;
cfg.channel = {'O1','Oz','O2'};
cfg.parameter = 'stat';
cfg.colormap = jet;
cfg.xlim = [0 0.6];
cfg.ylim = [13 40];
cfg.renderer  = 'painters';
cfg.marker = 'on';
cfg.style = 'fill';
cfg.comment = 'off';
cfg.maskparameter = 'mask';
cfg.maskstyle = 'outline';
cfg.colorbar = 'yes';
cfg.zlim = [-2.5 2.5];
figure;
ft_singleplotTFR(cfg,stat05);
set(gca, 'Fontsize',20);
title ('Mean over visual');
set(gca,'box','on');
xlabel('time (s)');
ylabel('frequency (Hz)');
c = colorbar;
c.LineWidth = 1;
c.FontSize  = 18;
title(c,'t-val');

% using no correction for multiple comparisons
cfg.correctm          = 'no';
statpar_no = ft_freqstatistics(cfg, bslnTimeFreqAvg_Con{:},bslnTimeFreqAvg_Incon{:});

% using Bonferroni correction for multiple comparisons
cfg.correctm          = 'bonferroni';
statpar_bf = ft_freqstatistics(cfg,bslnTimeFreqAvg_Con{:},bslnTimeFreqAvg_Incon{:});

% using False Discovery Rate correction for multiple comparisons
cfg.correctm          = 'fdr';
statpar_fdr = ft_freqstatistics(cfg, bslnTimeFreqAvg_Con{:},bslnTimeFreqAvg_Incon{:});


statpar_no.effect = effect;

cfg = [];
cfg.channel       = {'C3','Cz','C4'};
cfg.baseline      = [-inf 0];
cfg.xlim = [0 0.6];
cfg.renderer      = 'openGL';     % painters does not support opacity, openGL does
cfg.colorbar      = 'yes';
cfg.parameter     = 'prob';       % display the power
cfg.maskparameter = 'mask';       % use significance to mask the power
cfg.maskalpha     = 0.3;          % make non-significant regions 30% visible
cfg.zlim          = 'maxabs';
figure
ft_singleplotTFR(cfg, statpar_no);

title('significant power changes (p<0.05, not corrected)')]


cfg = [];
cfg.channel       = {'O1','Oz','O2'};
cfg.baseline      = [-inf 0];
cfg.xlim = [0 0.6];
cfg.renderer      = 'openGL';     % painters does not support opacity, openGL does
cfg.colorbar      = 'yes';
cfg.parameter     = 'prob';       % display the power
cfg.maskparameter = 'mask';       % use significance to mask the power
cfg.maskalpha     = 0.3;          % make non-significant regions 30% visible
cfg.zlim          = 'maxabs';
figure
ft_singleplotTFR(cfg, statpar_no);

title('significant power changes (p<0.05, not corrected)')


% Define the configuration structure

%% chan_freq_time with singleton freq
stat.time = 1:97;
stat.freq = 1;
stat.dimord = 'chan_time';

%% chan_time, this is the original one
% insert the singleton dimension in the data
stat.posclusterslabelmat = reshape(stat.posclusterslabelmat, [32 3 97]);
stat.negclusterslabelmat = reshape(stat.negclusterslabelmat, [32 3 97]);
stat.prob     = reshape(stat.prob,    [32 3 97]);
stat.cirange  = reshape(stat.cirange, [32 3 97]);
stat.mask     = reshape(stat.mask,    [32 3 97]);
stat.stat     = reshape(stat.stat,    [32 3 97]);
stat.ref      = reshape(stat.ref,     [32 3 97]);

cfg.zlim       = [-4 4]; % set the color limits
cfg.highlight  = 'on';
cfg.highlightchannel = find(stat.mask);
cfg.highlightsymbol = 'o';
cfg.highlightcolor = [1 1 1];
cfg.highlightsize      = 10;
cfg.opacitymap  = 'rampup';  

ft_clusterplot(cfg,stat);


cfg = [];

cfg.highlightsymbolseries = ['*','*','.','.','.'];
cfg.layout = 'D:\Study\eeg data\time-frequency analysis\easycapM1.mat';
cfg.contournum = 0;
cfg.markersymbol = '.';
cfg.alpha = 0.05;
cfg.parameter='stat';
cfg.zlim = [-5 5];

ft_clusterplot(cfg,stat);


cfg = [];
cfg.alpha = 0.05;
cfg.parameter = 'raweffect';
cfg.zlim = [-4 4];
cfg.renderer      = 'painters'
cfg.layout = 'D:\Study\eeg data\time-frequency analysis\easycapM1.mat';
cfg.colorbar = 'yes';

stat_freq = squeeze(nanmean(stat(:,[2 7],:),2)); % average over frequency dimension

ft_clusterplot(cfg,stat_freq);



% Define the configuration structure
cfg = [];
cfg.alpha = 0.05;
cfg.parameter = 'stat';
cfg.zlim = [-4 4];
cfg.layout = 'D:\Study\eeg data\time-frequency analysis\easycapM1.mat';
cfg.colorbar = 'yes';

% Find indices of frequency range to average over
timewin = [36 38];
time = stat.time;
timewin_idx = find(time >= timewin(1) & time <= timewin(2));

% Average over frequency dimension using nanmean to handle NaN values
stat_freq = squeeze(nanmean(stat(:,:,timewin_idx), 3));

% Plot the results using ft_clusterplot
ft_clusterplot(cfg, stat_freq);


cfg = [];
cfg.channel          = {'all'};
cfg.latency          = 'all';
cfg.frequency        = [8 50];
cfg                  = [];
cfg.method           = 'montecarlo'; % use the Monte Carlo Method to calculate the significance probability
cfg.statistic        = 'indepsamplesT'; % use the independent samples T-statistic as a measure to
                                   % evaluate the effect at the sample level

cfg.correctm         = 'cluster';
cfg.clusteralpha     = 0.05;
cfg.clusterstatistic = 'maxsum';
cfg.minnbchan        = 2;
cfg.tail             = 0;
cfg.clustertail      = 0;
cfg.alpha            = 0.025;
cfg.numrandomization = 500;
cfg.design = design;
cfg.neighbours = neighbours;
cfg.uvar = 1; % row in design indicating subjects, repeated measure
cfg.ivar = 2; % row in design indicating condition for contrast

[stat] = ft_freqstatistics(cfg,bslnTimeFreqAvg_Con{:},bslnTimeFreqAvg_Incon{:});

cfg = [];

stat.raweffect = gAvgBslnTimeFreq_Con.powspctrm - gAvgBslnTimeFreq_Incon.powspctrm;


% Preparing the design matrix for the statistical evaluation
% For within-subjects analysis, the design matrix contains two rows

Subj= {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'};	 

Ns = length(Subj);

% cfg.design    = zeros(1, size(allsub,2));

cfg.design(1,1:2*Ns)  = [ones(1,Ns) 2*ones(1,Ns)];
cfg.design(2,1:2*Ns)  = [1:Ns 1:Ns];
cfg.ivar              = 1; % the 1st row in cfg.design contains the independent variable
cfg.uvar              = 2; % the 2nd row in cfg.design contains the subject number

%proceed with three methods that do correct for the MCP.
cfg.method    = 'analytic';
cfg.correctm  = 'no';
TFR_stat1 = ft_freqstatistics(cfg, gAvgBslnTimeFreq_Con);

cfg.method    = 'analytic';
cfg.correctm  = 'no';
TFR_stat1 = ft_freqstatistics(cfg, gAvgBslnTimeFreq_Incon);

cfg = [];
cfg.operation = 'subtract';
cfg.parameter = 'pow'; %'pow' or 'avg'

I-C = ft_math(cfg,bslnTimeFreqAvg_Con, bslnTimeFreqAvg_Incon); 
%A_vs_B = tfA_grand;
I-C.powspctrm = bslnTimeFreqAvg_Con.powspctrm - bslnTimeFreqAvg_Incon.powspctrm;

cfg.method            = 'montecarlo';
cfg.clusterthreshold = 'nonparametric_common';
cfg.correctm = 'cluster';　% correct for multiple comparisons
                         % 'no', 'max', 'cluster', 'bonferoni', holms, fdr

cfg.clusteralpha = 0.05;
cfg.clusterstatistic = 'maxsum'; %  maxsize’, or ‘wcm’
cfg.minnbchan = 2;
cfg.neighbours = neighbours;
cfg.numrandomization  = 1000; % 1000 is recommended, but takes longer


TFR_stat = ft_freqstatistics(cfg, I-C);
ft_multiplotTFR(cfg,TFR_stat);

stat = ft_freqstatistics(cfg,bslnTimeFreqAvg_Con{:},bslnTimeFreqAvg_Incon{:});

%% correction
% bonferroni correction
cfg.method    = 'analytic';
cfg.correctm  = 'bonferroni';
TFR_stat2     = ft_freqstatistics(cfg, stat);

% fdr correction   比较严格
cfg.method    = 'analytic';
cfg.correctm  = 'fdr';
TFR_stat3     = ft_freqstatistics(cfg, stat);

save('tf_result.mat','stat');


%% Visualise the results
cfg               = [];
cfg.marker        = 'on';
cfg.layout        = 'D:\Study\eeg data\time-frequency analysis\easycapM1.mat';
cfg.channel       = 'all';
cfg.parameter     = 'stat';  % plot the t-value
cfg.maskparameter = 'mask';  % use the thresholded probability to mask the data
% cfg.maskstyle     = 'outline';
cfg.maskalpha     = 0.05
cfg.xlim   = [-0.3 0.6];   % time limit, in second   
cfg.ylim   = [10 40];       % freq limit, in Hz   
cfg.zlim   = [-3 3];
cfg.showlabels   = 'yes';

figure; ft_multiplotTFR(cfg,TFR_stat);
figure; ft_multiplotTFR(cfg,gAvgBslnTimeFreq_Con);
figure; ft_multiplotTFR(cfg,gAvgBslnTimeFreq_Incon);
figure; ft_multiplotTFR(cfg,statTimeFreq);
figure; ft_multiplotTFR(cfg,stat01);
figure; ft_multiplotTFR(cfg,stat05);

%差异图
cfg = [];
cfg.xlim         = [-0.3 0.6];
cfg.zlim         = [-3 3];
cfg.showlabels   = 'yes';
cfg.showoutline   = 'yes';
cfg.layout       = 'D:\Study\eeg data\time-frequency analysis\easycapM1.mat';
ft_topoplotER(cfg,MC_M_diff);
diff = {'I-C', 'I-B'}
for g = 1:2
    diff_pingjun = eval(diff{g})
    figure
    ft_multiplotTFR(cfg,diff_pingjun);
    colormap(jet);
    colorbar;
    colormap(jet);
    title(diff{g})
end



%%further analysis
%get figure of grand average，choose time window, frequecy and interest
%channels
time  = gAvgBslnTimeFreq_Con.time;
freq  = gAvgBslnTimeFreq_Con.freq;
chan  = gAvgBslnTimeFreq_Con.label;

% define time window
timewin      = [0.2 0.3];
timewin_idx  = dsearchn(time', timewin');
% define frequency window
freqwin      = [13 30];  % theta band
freqwin_idx  = dsearchn(freq', freqwin');
% define ROI (channels)
chan2use = {'C3','Cz','C4'};
chan_idx = zeros(1,length(chan2use));

for i=1:length(chan2use)       % find the index of channels to use
    ch = strcmpi(chan2use(i), chan);
    chan_idx(i) = find(ch);
end

% extract mean TFR over these time window, freq window, and ROI, for each condition and each subject
Subj= {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'};	 

Ns = length(Subj);

power = zeros(Ns,2); % initialize variable， 此例中，有Ns个被试，2个条件

for i=1:Ns
    pow1 = bslnTimeFreqAvg_Con{1,i}.powspctrm(chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    pow2 = bslnTimeFreqAvg_Incon{1,i}.powspctrm( chan_idx, ...
        freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    %     pow2 = allsub_B{1,i}.powspctrm( chan_idx, ...
    %         freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    %     pow3 = allsub_C{1,i}.powspctrm( chan_idx, ...
    %         freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    %     pow4 = allsub_D{1,}.powspctrm( chan_idx, ...
    %         freqwin_idx(1):freqwin_idx(2), timewin_idx(1):timewin_idx(2) );
    power(i,1) = squeeze(mean(mean(mean( pow1 ))));  % 提取第一个条件的数据
    power(i,2) = squeeze(mean(mean(mean( pow2  ))));  % 提取第二个条件的数据
   
    %     power(subi,3) = squeeze(mean(mean(mean( pow3  ))));  % 提取第一个条件的数据
    %     power(subi,4) = squeeze(mean(mean(mean( pow4  ))));  % 提取第二个条件的数据
end

dlmwrite('D:\Study\eeg data\power.txt',power,'\t')  % 保存到txt文件中(用excel打开)，用于进一步分析
% 如有其他时频窗口或ROI，也类似操作
% 当选择了多个时频窗口或ROI时，进行了多次比较，此时需要对p值进行校正（eg, FDR）

%%ft_definetrial




 