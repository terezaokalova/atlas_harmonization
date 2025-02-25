function varargout = process_fooof(varargin)
% PROCESS_FOOOF: Applies the "Fitting Oscillations and One Over F" (specparam) algorithm on a Welch's PSD
%
% REFERENCE: Please cite the original algorithm:
%    Donoghue T, Haller M, Peterson E, Varma P, Sebastian P, Gao R, Noto T,
%    Lara AH, Wallis JD, Knight RT, Shestyuk A, Voytek B. Parameterizing 
%    neural power spectra into periodic and aperiodic components. 
%    Nature Neuroscience (2020)

% @=============================================================================
% This function is part of the Brainstorm software:
% https://neuroimage.usc.edu/brainstorm
% 
% Copyright (c) University of Southern California & McGill University
% This software is distributed under the terms of the GNU General Public License
% as published by the Free Software Foundation. Further details on the GPLv3
% license can be found at http://www.gnu.org/copyleft/gpl.html.
% 
% FOR RESEARCH PURPOSES ONLY. THE SOFTWARE IS PROVIDED "AS IS," AND THE
% UNIVERSITY OF SOUTHERN CALIFORNIA AND ITS COLLABORATORS DO NOT MAKE ANY
% WARRANTY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR DO THEY ASSUME ANY
% LIABILITY OR RESPONSIBILITY FOR THE USE OF THIS SOFTWARE.
%
% For more information type "brainstorm license" at command prompt.
% =============================================================================@
%
% Authors: Luc Wilson, Francois Tadel, 2020-2024

% --- If you're NOT in Brainstorm environment, 'macro_method' won't exist ---
if exist('macro_method','var') || exist('macro_method','file')
    eval(macro_method);
end

end


%% ===== GET DESCRIPTION =====
function sProcess = GetDescription() %#ok<DEFNU>
    % Description the process
    sProcess.Comment     = 'specparam: Fitting oscillations and 1/f';
    sProcess.Category    = 'Custom';
    sProcess.SubGroup    = 'Frequency';
    sProcess.Index       = 490;
    sProcess.Description = 'https://neuroimage.usc.edu/brainstorm/Tutorials/Fooof';
    % Definition of the input accepted by this process
    sProcess.InputTypes  = {'timefreq'};
    sProcess.OutputTypes = {'timefreq'};
    sProcess.nInputs     = 1;
    sProcess.nMinFiles   = 1;
    % Definition of the options
    % === FOOOF TYPE
    sProcess.options.implementation.Comment = {'Matlab', 'Python 3 (3.7 recommended)', 'specparam implementation:'; 'matlab', 'python', ''};
    sProcess.options.implementation.Type    = 'radio_linelabel';
    sProcess.options.implementation.Value   = 'matlab';
    sProcess.options.implementation.Controller.matlab = 'Matlab';
    sProcess.options.implementation.Controller.python = 'Python';
    % === FREQUENCY RANGE
    sProcess.options.freqrange.Comment = 'Frequency range for analysis: ';
    sProcess.options.freqrange.Type    = 'freqrange_static';
    sProcess.options.freqrange.Value   = {[1 40], 'Hz', 1};
    % === POWER LINE
    sProcess.options.powerline.Comment = {'None', '50 Hz', '60 Hz', 'Ignore power line frequencies:'; '-5', '50', '60', ''};
    sProcess.options.powerline.Type    = 'radio_linelabel';
    sProcess.options.powerline.Value   = 'None';
    sProcess.options.powerline.Class   = 'Matlab';
    % === MODEL SELECTION
    sProcess.options.method.Comment = {'Default', 'Model selection (experimental)', 'Optimization method:'; 'leastsquare', 'negloglike', ''};
    sProcess.options.method.Type    = 'radio_linelabel';
    sProcess.options.method.Value   = 'leastsquare';
    sProcess.options.method.Class   = 'Matlab';
    % === PEAK WIDTH LIMITS
    sProcess.options.peakwidth.Comment = 'Peak width limits (default=[0.5-12]): ';
    sProcess.options.peakwidth.Type    = 'freqrange_static';
    sProcess.options.peakwidth.Value   = {[0.5 12], 'Hz', 1};
    % === MAX PEAKS
    sProcess.options.maxpeaks.Comment = 'Maximum number of peaks (default=3): ';
    sProcess.options.maxpeaks.Type    = 'value';
    sProcess.options.maxpeaks.Value   = {3, '', 0};
    % === MEAN PEAK HEIGHT
    sProcess.options.minpeakheight.Comment = 'Minimum peak height (default=3): ';
    sProcess.options.minpeakheight.Type    = 'value';
    sProcess.options.minpeakheight.Value   = {3, 'dB', 1};
    % === PROXIMITY THRESHOLD
    sProcess.options.proxthresh.Comment = 'Proximity threshold (default=2): ';
    sProcess.options.proxthresh.Type    = 'value';
    sProcess.options.proxthresh.Value   = {2, 'stdev of peak model', 2};
    sProcess.options.proxthresh.Class   = 'Matlab';
    % === APERIODIC MODE 
    sProcess.options.apermode.Comment = {'Fixed', 'Knee', 'Aperiodic mode (default=fixed):'; 'fixed', 'knee', ''};
    sProcess.options.apermode.Type    = 'radio_linelabel';
    sProcess.options.apermode.Value   = 'fixed';
    % === GUESS WEIGHT
    sProcess.options.guessweight.Comment = {'None', 'Weak', 'Strong', 'Guess weight (default=none):'; 'none', 'weak', 'strong', ''};
    sProcess.options.guessweight.Type    = 'radio_linelabel';
    sProcess.options.guessweight.Value   = 'none';
    sProcess.options.guessweight.Class   = 'Matlab';
    
    % === SORT PEAKS TYPE
    sProcess.options.sorttype.Comment = {'Peak parameters', 'Frequency bands', 'Sort peaks using:'; 'param', 'band', ''};
    sProcess.options.sorttype.Type    = 'radio_linelabel';
    sProcess.options.sorttype.Value   = 'param';
    sProcess.options.sorttype.Controller.param = 'Param';
    sProcess.options.sorttype.Controller.band = 'Band';
    sProcess.options.sorttype.Group   = 'output';
    % === SORT PEAKS PARAM
    sProcess.options.sortparam.Comment = {'Frequency', 'Amplitude', 'Std dev.', 'Sort by peak...'; 'frequency', 'amplitude', 'std', ''};
    sProcess.options.sortparam.Type    = 'radio_linelabel';
    sProcess.options.sortparam.Value   = 'frequency';
    sProcess.options.sortparam.Class   = 'Param';
    sProcess.options.sortparam.Group   = 'output';
    % === SORT FREQ BANDS
    DefaultFreqBands = bst_get('DefaultFreqBands');
    sProcess.options.sortbands.Comment = '';
    sProcess.options.sortbands.Type    = 'groupbands';
    sProcess.options.sortbands.Value   = DefaultFreqBands(:,1:2);
    sProcess.options.sortbands.Class   = 'Band';
    sProcess.options.sortbands.Group   = 'output';
end


%% ===== FORMAT COMMENT =====
function Comment = FormatComment(sProcess) %#ok<DEFNU>
    Comment = sProcess.Comment;
end


%% ===== RUN =====
function OutputFile = Run(sProcess, sInputs) %#ok<DEFNU>
    % Initialize returned list of files
    OutputFile = {};
    
    % Fetch user settings
    implementation = sProcess.options.implementation.Value;
    opt.freq_range          = sProcess.options.freqrange.Value{1};
    opt.peak_width_limits   = sProcess.options.peakwidth.Value{1};
    opt.max_peaks           = sProcess.options.maxpeaks.Value{1};
    opt.min_peak_height     = sProcess.options.minpeakheight.Value{1} / 10; % convert from dB to B
    opt.aperiodic_mode      = sProcess.options.apermode.Value;
    opt.peak_threshold      = 2;   % 2 std dev
    opt.border_threshold    = 1;   % 1 std dev
    opt.return_spectrum     = 0;
    opt.power_line          = sProcess.options.powerline.Value;
    opt.proximity_threshold = sProcess.options.proxthresh.Value{1};
    opt.optim_obj           = sProcess.options.method.Value; 
    opt.peak_type           = 'gaussian'; 
    opt.guess_weight        = sProcess.options.guessweight.Value;
    opt.thresh_after        = true; 
    opt.verbose             = false;
    opt.sort_type           = sProcess.options.sorttype.Value;
    opt.sort_param          = sProcess.options.sortparam.Value;
    opt.sort_bands          = sProcess.options.sortbands.Value;

    if (any(opt.freq_range < 0) || opt.freq_range(1) >= opt.freq_range(2))
        bst_report('error','Invalid Frequency range');
        return
    end
    
    hasOptimTools = 0;
    if exist('fmincon','file') == 2 && strcmp(implementation,'matlab')
        hasOptimTools = 1;
        disp('Using constrained optimization, Guess Weight ignored.')
    end
    
    OutputFile = {};
    for iFile = 1:length(sInputs)
        bst_progress('text',['Standby: FOOOFing spectrum ' num2str(iFile) ' of ' num2str(length(sInputs))]);
        % Load input file
        PsdMat = in_bst_timefreq(sInputs(iFile).FileName);
        % Exclude 0Hz
        if (opt.freq_range(1) == 0) && (PsdMat.Freqs(1) == 0) && (length(PsdMat.Freqs) >= 2)
            opt.freq_range(1) = PsdMat.Freqs(2);
        end
        
        % Switch between implementations
        switch (implementation)
            case 'matlab'
                switch (opt.optim_obj)
                    case 'leastsquare'
                        [FOOOF_freqs, FOOOF_data, errMsg] = FOOOF_matlab(PsdMat.TF, PsdMat.Freqs, opt, hasOptimTools);
                    case 'negloglike'
                        [FOOOF_freqs, FOOOF_data, errMsg] = FOOOF_matlab_nll(PsdMat.TF, PsdMat.Freqs, opt, hasOptimTools);
                end
            case 'python'
                opt.peak_type = 'gaussian';
                opt.optim_obj = 'leastsquare';
                [FOOOF_freqs, FOOOF_data, errMsg] = process_fooof_py('FOOOF_python', PsdMat.TF, PsdMat.Freqs, opt);
                FOOOF_data = [FOOOF_data.FOOOF];
            otherwise
                errMsg = ['Invalid FOOOF implentation: ' implementation];
        end
        if ~isempty(errMsg)
            bst_report('Error', sProcess, sInputs(iFile), errMsg);
            return;
        end

        TFfooof = PsdMat.TF(:,1,ismember(PsdMat.Freqs,FOOOF_freqs));
        [ePeaks, eAperiodics, eStats] = FOOOF_analysis(FOOOF_data, PsdMat.RowNames, TFfooof, opt.max_peaks, opt.sort_type, opt.sort_param, opt.sort_bands); 
        
        PsdMat.Options.FOOOF = struct(...
            'options',    opt, ...
            'freqs',      FOOOF_freqs, ...
            'data',       FOOOF_data, ...
            'peaks',      ePeaks, ...
            'aperiodics', eAperiodics, ...
            'stats',      eStats);
        mstag = '';
        if ~isempty(strfind(opt.optim_obj, 'negloglike'))
            mstag = 'ms-';
        end
        if ~isempty(strfind(PsdMat.Comment, 'PSD:'))
            PsdMat.Comment = strrep(PsdMat.Comment, 'PSD:', [mstag 'specparam:']);
        else
            PsdMat.Comment = strcat(PsdMat.Comment, [' | ' mstag 'specparam']);
        end
        PsdMat = bst_history('add', PsdMat, 'compute', 'specparam');
        
        [fPath, fName, fExt] = bst_fileparts(file_fullpath(sInputs(iFile).FileName));
        NewFile = file_unique(bst_fullfile(fPath, [fName, '_specparam', fExt]));
        bst_save(NewFile, PsdMat, 'v6');
        db_add_data(sInputs(iFile).iStudy, NewFile, PsdMat);
        OutputFile{end+1} = NewFile;
    end
end


%% ===================================================================================
%  ===== MATLAB FOOOF ================================================================
%  ===================================================================================
function [fs, fg, errMsg] = FOOOF_matlab_nll(TF, Freqs, opt, hOT)
    errMsg = '';
    fMask = (round(Freqs.*10)./10 >= opt.freq_range(1)) & ...
            (round(Freqs.*10)./10 <= opt.freq_range(2)) & ...
            ~mod(sum(abs(round(Freqs.*10)./10-[1;2;3].*str2double(opt.power_line)) >= 2),3);
    fs = Freqs(fMask);
    spec = log10(squeeze(TF(:,1,fMask)));
    nChan = size(TF,1);
    if nChan == 1, spec = spec'; end
    
    fg(nChan) = struct('aperiodic_params', [], 'peak_params', [], ...
            'peak_types', '', 'ap_fit', [], 'fooofed_spectrum', [], ...
            'peak_fit', [], 'error', [], 'r_squared', []);
    
    bst_progress('text','Standby: ms-specparam is running in parallel');
    try
        parfor chan = 1:nChan
            % Pre-initialize variables to avoid "uninitialized temporaries":
            lb = []; 
            ub = [];
            aperiodic_pars_out = [];
            pk_pars_out = [];
            
            bst_progress('set', bst_round(chan / nChan,2) * 100);
            aperiodic_pars = robust_ap_fit(fs, spec(chan,:), opt.aperiodic_mode);
            flat_spec = flatten_spectrum(fs, spec(chan,:), aperiodic_pars, opt.aperiodic_mode);
            
            [est_pars, peak_function] = est_peaks(fs, flat_spec, opt.max_peaks, ...
                opt.peak_threshold, opt.min_peak_height, opt.peak_width_limits/2, ...
                opt.proximity_threshold, opt.border_threshold, opt.peak_type);
            
            nPk = size(est_pars,1);
            % Pre-allocate array of models
            model(nPk+1) = struct('aperiodic_params', [], 'peak_params', [], ...
                                  'MSE', [], 'r_squared', [], 'loglik', [], ...
                                  'AIC', [], 'BIC', [], 'BF', []);

            for pk = 0:nPk
                peak_pars = est_fit(est_pars(1:pk,:), fs, flat_spec, ...
                    opt.peak_width_limits/2, opt.peak_type, opt.guess_weight, hOT);
                
                aperiodic = spec(chan,:);
                for peakIdx = 1:size(peak_pars,1)
                    aperiodic = aperiodic - peak_function(fs,peak_pars(peakIdx,1), ...
                                          peak_pars(peakIdx,2), peak_pars(peakIdx,3));
                end
                aperiodic_pars_temp = simple_ap_fit(fs, aperiodic, opt.aperiodic_mode);
                guess = [aperiodic_pars_temp'; peak_pars(:)];
                
                switch opt.aperiodic_mode
                    case 'fixed'
                        lb_ap = [-inf; 0];
                        ub_ap = [ inf; inf];
                    case 'knee'
                        lb_ap = [-inf; 0; 0];
                        ub_ap = [ inf; 100; inf];
                end
                if ~isempty(peak_pars)
                    lb_pk = [];
                    ub_pk = [];
                    for ip=1:size(peak_pars,1)
                        cfrLow = max(0,peak_pars(ip,1)-2*peak_pars(ip,3));
                        cfrHigh= peak_pars(ip,1)+2*peak_pars(ip,3);
                        lb_pk = [lb_pk; cfrLow; 0; opt.peak_width_limits(1)/2];
                        ub_pk = [ub_pk; cfrHigh; inf; opt.peak_width_limits(2)/2];
                    end
                    lb = [lb_ap; lb_pk];
                    ub = [ub_ap; ub_pk];
                else
                    lb = lb_ap;
                    ub = ub_ap;
                end
                
                options = optimset('Display','off','TolX',1e-7,'TolFun',1e-9, ...
                    'MaxFunEvals',5000,'MaxIter',5000);
                
                params = fmincon(@(p)err_fm_constr(p, fs, spec(chan,:), ...
                    opt.aperiodic_mode, opt.peak_type), guess, [], [], [], [], lb, ub, [], options);
                
                switch opt.aperiodic_mode
                    case 'fixed'
                        if length(params)>2
                            aperiodic_pars_out = params(1:2);
                            pk_pars_out = reshape(params(3:end),[3,numel(params(3:end))/3])';
                        else
                            aperiodic_pars_out = params(1:2);
                            pk_pars_out = zeros(1,3);
                        end
                    case 'knee'
                        if length(params)>3
                            aperiodic_pars_out = params(1:3);
                            pk_pars_out = reshape(params(4:end),[3,numel(params(4:end))/3])';
                        else
                            aperiodic_pars_out = params(1:3);
                            pk_pars_out = zeros(1,3);
                        end
                end
                
                ap_fit = gen_aperiodic(fs, aperiodic_pars_out, opt.aperiodic_mode);
                model_fit = ap_fit;
                if any(pk_pars_out(:)~=0)
                    for kk=1:size(pk_pars_out,1)
                        model_fit = model_fit + peak_function(fs,pk_pars_out(kk,1), ...
                                            pk_pars_out(kk,2),pk_pars_out(kk,3));
                    end
                else
                    pk_pars_out = zeros(1,3);
                end
                mseVal = sum((spec(chan,:)-model_fit).^2)/numel(model_fit);
                cc = corrcoef(spec(chan,:), model_fit);
                if size(cc,1)>1
                    r2val = cc(1,2)^2;
                else
                    r2val = 0;
                end
                loglik = -numel(model_fit)/2*(1+log(mseVal)+log(2*pi));
                AIC = 2*(length(params)-loglik);
                BIC = length(params)*log(numel(model_fit)) - 2*loglik;
                
                model(pk+1).aperiodic_params = aperiodic_pars_out;
                model(pk+1).peak_params      = pk_pars_out;
                model(pk+1).MSE              = mseVal;
                model(pk+1).r_squared        = r2val;
                model(pk+1).loglik           = loglik;
                model(pk+1).AIC              = AIC;
                model(pk+1).BIC              = BIC;
                % Compare to the "0 peak" BIC for BF
                model(pk+1).BF               = exp((BIC - model(1).BIC)/2);
            end
            
            allBIC=[model.BIC];
            [~,mi]=min(allBIC);
            bestMod=model(mi);
            bestMod.aperiodic_params(2)=abs(bestMod.aperiodic_params(2));
            fg(chan).aperiodic_params = bestMod.aperiodic_params;
            fg(chan).peak_params      = bestMod.peak_params;
            fg(chan).peak_types       = func2str(peak_function);
            fg(chan).ap_fit           = 10.^gen_aperiodic(fs,bestMod.aperiodic_params,opt.aperiodic_mode);
            finalModel=build_model(fs,bestMod.aperiodic_params,opt.aperiodic_mode,bestMod.peak_params,peak_function);
            fg(chan).fooofed_spectrum = 10.^finalModel;
            fg(chan).peak_fit         = fg(chan).fooofed_spectrum ./ fg(chan).ap_fit;
            fg(chan).error            = bestMod.MSE;
            fg(chan).r_squared        = bestMod.r_squared;
        end
    catch err
        errMsg = err.message;
    end
end


function [fs, fg, errMsg] = FOOOF_matlab(TF, Freqs, opt, hOT)
    errMsg='';
    fMask=(round(Freqs.*10)./10>=opt.freq_range(1)) & ...
          (round(Freqs.*10)./10<=opt.freq_range(2)) & ...
          ~mod(sum(abs(round(Freqs.*10)./10-[1;2;3].*str2double(opt.power_line))>=2),3);
    fs=Freqs(fMask);
    spec=log10(squeeze(TF(:,1,fMask)));
    nChan=size(TF,1);
    if nChan==1, spec=spec'; end
    fg(nChan)=struct('aperiodic_params',[],'peak_params',[], ...
                     'peak_types','','ap_fit',[], ...
                     'fooofed_spectrum',[],'peak_fit',[], ...
                     'error',[],'r_squared',[]);
    for chan=1:nChan
        bst_progress('set', round(chan./nChan.*100));
        aperiodic_pars=robust_ap_fit(fs,spec(chan,:),opt.aperiodic_mode);
        flat_spec=flatten_spectrum(fs,spec(chan,:),aperiodic_pars,opt.aperiodic_mode);
        [peak_pars,peak_function]=fit_peaks(fs,flat_spec,opt.max_peaks,opt.peak_threshold, ...
            opt.min_peak_height,opt.peak_width_limits/2,opt.proximity_threshold, ...
            opt.border_threshold,opt.peak_type,opt.guess_weight,hOT);
        
        if opt.thresh_after && ~hOT
            peak_pars(peak_pars(:,2)<opt.min_peak_height,:)=[];
            peak_pars(peak_pars(:,3)<opt.peak_width_limits(1)/2,:)=[];
            peak_pars(peak_pars(:,3)>opt.peak_width_limits(2)/2,:)=[];
            peak_pars=drop_peak_cf(peak_pars,opt.border_threshold,opt.freq_range);
            peak_pars(peak_pars(:,1)<0,:)=[];
            peak_pars=drop_peak_overlap(peak_pars,opt.proximity_threshold);
        end
        
        aperiodic=spec(chan,:);
        for pk=1:size(peak_pars,1)
            aperiodic=aperiodic-peak_function(fs,peak_pars(pk,1), ...
                             peak_pars(pk,2),peak_pars(pk,3));
        end
        aperiodic_pars=simple_ap_fit(fs,aperiodic,opt.aperiodic_mode);
        
        ap_fit=gen_aperiodic(fs,aperiodic_pars,opt.aperiodic_mode);
        model_fit=ap_fit;
        for pk=1:size(peak_pars,1)
            model_fit=model_fit+peak_function(fs,peak_pars(pk,1), ...
                               peak_pars(pk,2),peak_pars(pk,3));
        end
        MSE=sum((spec(chan,:)-model_fit).^2)/numel(model_fit);
        cc=corrcoef(spec(chan,:),model_fit);
        rsq_tmp=0;if size(cc,1)>1, rsq_tmp=cc(1,2)^2;end
        aperiodic_pars(2)=abs(aperiodic_pars(2));
        fg(chan).aperiodic_params=aperiodic_pars;
        fg(chan).peak_params=peak_pars;
        fg(chan).peak_types=func2str(peak_function);
        fg(chan).ap_fit=10.^ap_fit;
        fg(chan).fooofed_spectrum=10.^model_fit;
        fg(chan).peak_fit=10.^(model_fit-ap_fit);
        fg(chan).error=MSE;
        fg(chan).r_squared=rsq_tmp;
        if opt.return_spectrum
            fg(chan).power_spectrum=spec(chan,:);
        end
    end
end


%% ===== GENERATE APERIODIC =====
function ap_vals = gen_aperiodic(freqs,aperiodic_params,aperiodic_mode)
    switch aperiodic_mode
        case 'fixed'
            ap_vals = expo_nk_function(freqs,aperiodic_params);
        case 'knee'
            ap_vals = expo_function(freqs,aperiodic_params);
        case 'floor'
            ap_vals = expo_fl_function(freqs,aperiodic_params);
    end
end


%% ===== CORE MODELS =====
function ys = gaussian(freqs, mu, hgt, sigma)
    ys = hgt.*exp(-(((freqs-mu)./sigma).^2)./2);
end

function ys = cauchy(freqs, ctr, hgt, gam)
    ys = hgt./(1+((freqs-ctr)./gam).^2);
end

function ys = expo_function(freqs,params)
    ys = params(1)-log10(abs(params(2))+freqs.^params(3));
end

function ys = expo_nk_function(freqs,params)
    ys = params(1)-log10(freqs.^params(2));
end

function ys = expo_fl_function(f,params)
    ys=log10(f.^(params(1))*10^(params(2))+params(3));
end


%% ===== FITTING ALGORITHM =====
function aperiodic_params=simple_ap_fit(freqs,power_spectrum,aperiodic_mode)
    options=optimset('Display','off','TolX',1e-4,'TolFun',1e-6,...
        'MaxFunEvals',1e4,'MaxIter',1e4);
    switch aperiodic_mode
        case 'fixed'
            exp_guess=-(power_spectrum(end)-power_spectrum(1))./log10(freqs(end)./freqs(1));
            guess_vec=[power_spectrum(1),exp_guess];
            aperiodic_params=fminsearch(@error_expo_nk_function,guess_vec,options,freqs,power_spectrum);
        case 'knee'
            exp_guess=-(power_spectrum(end)-power_spectrum(1))./log10(freqs(end)./freqs(1));
            guess_vec=[power_spectrum(1),0,exp_guess];
            aperiodic_params=fminsearch(@error_expo_function,guess_vec,options,freqs,power_spectrum);
    end
end

function aperiodic_params=robust_ap_fit(freqs,power_spectrum,aperiodic_mode)
    popt=simple_ap_fit(freqs,power_spectrum,aperiodic_mode);
    initial_fit=gen_aperiodic(freqs,popt,aperiodic_mode);
    flatspec=power_spectrum-initial_fit;
    flatspec(flatspec<0)=0;
    perc_thresh=bst_prctile(flatspec,2.5);
    mask=flatspec<=perc_thresh;
    x2=freqs(mask);
    y2=power_spectrum(mask);
    options=optimset('Display','off','TolX',1e-4,'TolFun',1e-6,...
        'MaxFunEvals',1e4,'MaxIter',1e4);
    switch aperiodic_mode
        case 'fixed'
            aperiodic_params=fminsearch(@error_expo_nk_function,popt,options,x2,y2);
        case 'knee'
            aperiodic_params=fminsearch(@error_expo_function,popt,options,x2,y2);
    end
end

function spectrum_flat=flatten_spectrum(freqs,power_spectrum,robust_aperiodic_params,aperiodic_mode)
    spectrum_flat=power_spectrum-gen_aperiodic(freqs,robust_aperiodic_params,aperiodic_mode);
end

function [model_params,peak_function]=est_peaks(freqs,flat_iter,max_n_peaks,peak_threshold,min_peak_height,gauss_std_limits,proxThresh,bordThresh,peakType)
    switch peakType
        case 'gaussian'
            peak_function=@gaussian;
            guess_params=zeros(max_n_peaks,3);
            specNow=flat_iter;
            df=freqs(2)-freqs(1);
            for i=1:max_n_peaks
                [mxVal, idx]=max(specNow);
                if mxVal<peak_threshold*std(specNow)||mxVal<min_peak_height
                    break
                end
                ctrFreq=freqs(idx);
                halfht=0.5*mxVal;
                leftIdx=idx;while (leftIdx>1)&&(specNow(leftIdx)>halfht),leftIdx=leftIdx-1;end
                rightIdx=idx;while (rightIdx<length(specNow))&&(specNow(rightIdx)>halfht),rightIdx=rightIdx+1;end
                fwhm=(rightIdx-leftIdx)*df;
                guess_std=fwhm/(2*sqrt(2*log(2)));
                if guess_std<gauss_std_limits(1),guess_std=gauss_std_limits(1);end
                if guess_std>gauss_std_limits(2),guess_std=gauss_std_limits(2);end
                guess_params(i,:)=[ctrFreq,mxVal,guess_std];
                specNow=specNow-gaussian(freqs,ctrFreq,mxVal,guess_std);
            end
            guess_params(~any(guess_params,2),:)=[];
            guess_params=drop_peak_cf(guess_params,bordThresh,[min(freqs) max(freqs)]);
            guess_params=drop_peak_overlap(guess_params,proxThresh);
            model_params=guess_params;
        case 'cauchy'
            peak_function=@cauchy;
            guess_params=zeros(max_n_peaks,3);
            specNow=flat_iter;
            df=freqs(2)-freqs(1);
            for i=1:max_n_peaks
                [mxVal,idx]=max(specNow);
                if mxVal<peak_threshold*std(specNow)||mxVal<min_peak_height
                    break
                end
                ctrFreq=freqs(idx);
                halfht=0.5*mxVal;
                leftIdx=idx;while (leftIdx>1)&&(specNow(leftIdx)>halfht),leftIdx=leftIdx-1;end
                rightIdx=idx;while (rightIdx<length(specNow))&&(specNow(rightIdx)>halfht),rightIdx=rightIdx+1;end
                fwhm=(rightIdx-leftIdx)*df;
                guess_gamma=fwhm/2;
                if guess_gamma<gauss_std_limits(1),guess_gamma=gauss_std_limits(1);end
                if guess_gamma>gauss_std_limits(2),guess_gamma=gauss_std_limits(2);end
                guess_params(i,:)=[ctrFreq,mxVal,guess_gamma];
                specNow=specNow-cauchy(freqs,ctrFreq,mxVal,guess_gamma);
            end
            guess_params(~any(guess_params,2),:)=[];
            guess_params=drop_peak_cf(guess_params,proxThresh,[min(freqs) max(freqs)]);
            guess_params=drop_peak_overlap(guess_params,proxThresh);
            model_params=guess_params;
    end
end

function guess=drop_peak_cf(guess,bw_std_edge,freq_range)
    cf_params=guess(:,1)';
    bw_params=guess(:,3)'*bw_std_edge;
    keep_peak=(abs(cf_params-freq_range(1))>bw_params)&(abs(cf_params-freq_range(2))>bw_params);
    guess=guess(keep_peak,:);
end

function guess=drop_peak_overlap(guess,thr)
    if size(guess,1)<2,return;end
    guess=sortrows(guess);
    bounds=[guess(:,1)-guess(:,3)*thr, guess(:,1), guess(:,1)+guess(:,3)*thr];
    drop_inds=[];
    for ind=1:size(bounds,1)-1
        b0=bounds(ind,:);
        b1=bounds(ind+1,:);
        if b0(2)>b1(1)
            if guess(ind,2)<guess(ind+1,2)
                drop_inds=[drop_inds ind];
            else
                drop_inds=[drop_inds ind+1];
            end
        end
    end
    guess(drop_inds,:)=[];
    guess=sortrows(guess,2,'descend');
end

function best_params=est_fit(guess_params,freqs,flat_spec,gauss_std_limits,peakType,guess_weight,hOT)
    if isempty(guess_params)
        best_params=zeros(0,3);
        return;
    end
    best_params=fit_peak_guess(guess_params,freqs,flat_spec,(peakType=='cauchy')+1,guess_weight,gauss_std_limits,hOT);
end

function peak_params=fit_peak_guess(guess,freqs,flat_spec,peak_type,guess_weight,std_limits,hOT)
    if hOT
        lb=[max([freqs(1)*ones(size(guess,1),1), guess(:,1)-2*guess(:,3)],[],2), ...
            zeros(size(guess,1),1), std_limits(1)*ones(size(guess,1),1)];
        ub=[min([freqs(end)*ones(size(guess,1),1), guess(:,1)+2*guess(:,3)],[],2), ...
            inf(size(guess,1),1), std_limits(2)*ones(size(guess,1),1)];
        options=optimset('Display','off','TolX',1e-3,'TolFun',1e-5,'MaxFunEvals',5e3,'MaxIter',5e3);
        peak_params=fmincon(@error_model_constr,guess,[],[],[],[],lb,ub,[],options,freqs,flat_spec,peak_type);
    else
        options=optimset('Display','off','TolX',1e-5,'TolFun',1e-7,'MaxFunEvals',5e3,'MaxIter',5e3);
        peak_params=fminsearch(@error_model,guess,options,freqs,flat_spec,peak_type,guess,guess_weight);
    end
end

function err=error_model(params,xVals,yVals,peak_type,guess,guess_weight)
    fitted_vals=0;weak=1e2;strong=1e7;
    for i=1:size(params,1)
        switch peak_type
            case 1
                fitted_vals=fitted_vals+gaussian(xVals,params(i,1),params(i,2),params(i,3));
            case 2
                fitted_vals=fitted_vals+cauchy(xVals,params(i,1),params(i,2),params(i,3));
        end
    end
    switch guess_weight
        case 'none'
            err=sum((yVals-fitted_vals).^2);
        case 'weak'
            err=sum((yVals-fitted_vals).^2)+...
                weak*sum((params(:,1)-guess(:,1)).^2)+...
                weak*sum((params(:,2)-guess(:,2)).^2);
        case 'strong'
            err=sum((yVals-fitted_vals).^2)+...
                strong*sum((params(:,1)-guess(:,1)).^2)+...
                strong*sum((params(:,2)-guess(:,2)).^2);
    end
end

function err=error_model_constr(params,xVals,yVals,peak_type)
    fitted_vals=0;
    for i=1:size(params,1)
        switch peak_type
            case 1
                fitted_vals=fitted_vals+gaussian(xVals,params(i,1),params(i,2),params(i,3));
            case 2
                fitted_vals=fitted_vals+cauchy(xVals,params(i,1),params(i,2),params(i,3));
        end
    end
    err=sum((yVals-fitted_vals).^2);
end

function err=err_fm_constr(params,xVals,yVals,aperiodic_mode,peak_type)
    switch aperiodic_mode
        case 'fixed'
            npk=(length(params)-2)/3;
            fitted_vals=-log10(xVals.^params(2))+params(1);
        case 'knee'
            npk=(length(params)-3)/3;
            fitted_vals=params(1)-log10(abs(params(2))+xVals.^params(3));
    end
    for s=1:npk
        switch peak_type
            case 'gaussian'
                fitted_vals=fitted_vals+gaussian(xVals,params(3*s),params(3*s+1),params(3*s+2));
            case 'cauchy'
                fitted_vals=fitted_vals+cauchy(xVals,params(3*s),params(3*s+1),params(3*s+2));
        end
    end
    err=sum((yVals-fitted_vals).^2);
end


%% ===================================================================================
%  ===== FOOOF STATS =================================================================
%  ===================================================================================
function [ePeaks,eAper,eStats] = FOOOF_analysis(FOOOF_data,ChanNames,TF,max_peaks,sort_type,sort_param,sort_bands)
    nChan=numel(ChanNames);
    switch sort_type
        case 'param'
            ePeaks=struct('channel',[],'center_frequency',[],'amplitude',[],'std_dev',[]);
            i=0;
            for chan=1:nChan
                if ~isempty(FOOOF_data(chan).peak_params)
                    for p=1:size(FOOOF_data(chan).peak_params,1)
                        i=i+1;
                        ePeaks(i).channel=ChanNames(chan);
                        ePeaks(i).center_frequency=FOOOF_data(chan).peak_params(p,1);
                        ePeaks(i).amplitude=FOOOF_data(chan).peak_params(p,2);
                        ePeaks(i).std_dev=FOOOF_data(chan).peak_params(p,3);
                    end
                end
            end
            switch sort_param
                case 'frequency'
                    [~,iSort]=sort([ePeaks.center_frequency]);ePeaks=ePeaks(iSort);
                case 'amplitude'
                    [~,iSort]=sort([ePeaks.amplitude]);ePeaks=ePeaks(iSort(end:-1:1));
                case 'std'
                    [~,iSort]=sort([ePeaks.std_dev]);ePeaks=ePeaks(iSort);
            end
        case 'band'
            ePeaks=struct('channel',[],'center_frequency',[],'amplitude',[],'std_dev',[],'band',[]);
            bands=process_tf_bands('Eval',sort_bands);
            i=0;
            for chan=1:nChan
                if ~isempty(FOOOF_data(chan).peak_params)
                    for p=1:size(FOOOF_data(chan).peak_params,1)
                        i=i+1;
                        ePeaks(i).channel=ChanNames(chan);
                        ePeaks(i).center_frequency=FOOOF_data(chan).peak_params(p,1);
                        ePeaks(i).amplitude=FOOOF_data(chan).peak_params(p,2);
                        ePeaks(i).std_dev=FOOOF_data(chan).peak_params(p,3);
                        bandRanges=cell2mat(bands(:,2));
                        iBand=find(ePeaks(i).center_frequency>=bandRanges(:,1)&ePeaks(i).center_frequency<=bandRanges(:,2));
                        if ~isempty(iBand)
                            ePeaks(i).band=bands{iBand,1};
                        else
                            ePeaks(i).band='None';
                        end
                    end
                end
            end
    end
    
    hasKnee=length(FOOOF_data(1).aperiodic_params)-2;
    eAper=struct('channel',[],'offset',[],'exponent',[]);
    for c=1:nChan
        eAper(c).channel=ChanNames(c);
        eAper(c).offset=FOOOF_data(c).aperiodic_params(1);
        if hasKnee
            eAper(c).exponent=FOOOF_data(c).aperiodic_params(3);
            eAper(c).knee_frequency=FOOOF_data(c).aperiodic_params(2);
        else
            eAper(c).exponent=FOOOF_data(c).aperiodic_params(2);
        end
    end
    
    eStats=struct('channel',ChanNames);
    for c=1:nChan
        eStats(c).MSE=FOOOF_data(c).error;
        eStats(c).r_squared=FOOOF_data(c).r_squared;
        spec=squeeze(log10(TF(c,:,:)));
        fspec=squeeze(log10(FOOOF_data(c).fooofed_spectrum))';
        eStats(c).frequency_wise_error=abs(spec-fspec);
    end
end
