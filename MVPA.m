
% load condition data

load dataFC_LP
load dataIC_LP

% determine the number of trials


nFC = numel(dataFC_LP.trial);
nIC = numel(dataIC_LP.trial);


cfg = [] ;
cfg.method          = 'mvpa';
cfg.latency         = [0.5, 0.7];
%cfg.features         = 'chan'; %'time' to see where;  'chan' to see when
cfg.avgovertime     = 'yes';
cfg.design          = [ones(nFC,1); 2*ones(nIC,1)];
cfg.features        = 'chan';
cfg.mvpa            = [];
cfg.mvpa.classifier = 'multiclass_lda';
cfg.mvpa.metric     = 'accuracy';
cfg.mvpa.repeat      = 2;
cfg.mvpa.k          = 5; % or 10, the number of folds to train and test the cross-validated performance

cfg.neighbours  = neighbours;

stat = ft_timelockstatistics(cfg, dataFC_LP, dataIC_LP)

dat = ft_appenddata([], dataFC_LP, dataIC_LP);
stat = ft_timelockstatistics(cfg, dat);

fprintf('Classification accuracy: %0.2f\n', stat.accuracy)


cfg.mvpa.metric      = 'confusion';
stat = ft_timelockstatistics(cfg, dataFC_LP, dataIC_LP)

stat.confusion

mv_plot_result(stat.mvpa)

mv_plot_result(stat.mvpa, stat.time)

cfg              = [];
cfg.parameter    = 'accuracy';
cfg.layout       = 'CTF151_helmet.mat';
cfg.xlim         = [0, 0];
cfg.colorbar     = 'yes';
ft_topoplotER(cfg, stat);

%Search across both time and channels

cfg = [] ;
cfg.method        = 'mvpa';
cfg.latency       = [-0.1, 0.8];
cfg.features      = [];
cfg.mvpa.repeat   = 2;
cfg.design        = [ones(nFC,1); 2*ones(nIC,1)];
stat = ft_timelockstatistics(cfg, dataFC_LP, dataIC_LP)


mv_plot_result(stat.mvpa, stat.time)
set(gca, 'YTick', 1:2:length(stat.label), 'YTickLabel', stat.label(1:2:end))