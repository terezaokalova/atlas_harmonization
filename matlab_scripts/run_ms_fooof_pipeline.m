function run_ms_fooof_pipeline()
% run_ms_fooof_pipeline.m
%
% 1) Load HUP time-series (as in the catch22 pipeline)
% 2) Compute PSD (Welch)
% 3) Fit ms-specparam (negative log-likelihood + BIC) for each electrode
% 4) Save results to CSV
%
% No Brainstorm calls or parfor. No warnings about uninitialized variables.

    % -----------------------------------------------------------------
    % 1) EXACT PATHS
    % -----------------------------------------------------------------
    DATA_BASE_PATH  = '/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/Data';
    RESULTS_BASE    = '/Users/tereza/nishant/atlas/atlas_work_terez/atlas_harmonization/results';
    HUP_ATLAS_PATH  = fullfile(DATA_BASE_PATH, 'hup_atlas.mat');

    % -----------------------------------------------------------------
    % 2) LOAD HUP TIME-SERIES DATA
    % -----------------------------------------------------------------
    disp('Loading HUP time series...');
    tmp = load(HUP_ATLAS_PATH, 'wake_clip');
    if ~isfield(tmp, 'wake_clip')
        error('Expected variable "wake_clip" not found in %s', HUP_ATLAS_PATH);
    end
    data = tmp.wake_clip;  % (#samples x #electrodes)

    fs = 200;  % sampling freq
    [nRows, nCols] = size(data);
    if nRows < nCols
        warning('Data has (#electrodes > #samples). Transpose if needed.');
        % data = data.';  % uncomment if needed
        % [nRows, nCols] = size(data);
    end
    nElec = nCols;
    fprintf('Data loaded: %d samples x %d electrodes.\n', nRows, nCols);

    % -----------------------------------------------------------------
    % 3) COMPUTE PSD (1-40 Hz) VIA WELCH FOR EACH ELECTRODE
    % -----------------------------------------------------------------
    disp('Computing PSD (Welch)...');
    nfft    = fs;         % 1-second window
    overlap = 0.5;        % 50%
    window  = hamming(nfft);

    allPsds  = cell(nElec,1);
    allFreqs = [];
    for e = 1:nElec
        x = data(:,e);
        [pxx,faxis] = pwelch(x, window, round(overlap*nfft), nfft, fs, 'psd');
        % restrict to 1-40 Hz
        keepIdx = (faxis >= 1 & faxis <= 40);
        faxis   = faxis(keepIdx);
        pxx     = pxx(keepIdx);
        % log10
        logPxx  = log10(pxx);

        allPsds{e} = logPxx;
        if isempty(allFreqs)
            allFreqs = faxis;
        end
    end
    fprintf('Finished computing PSD for %d electrodes.\n', nElec);

    % -----------------------------------------------------------------
    % 4) RUN MS-SPECPARAM (NLL + BIC) ON EACH ELECTRODE
    % -----------------------------------------------------------------
    disp('Fitting ms-specparam (NLL + BIC) per electrode...');

    freqRange        = [1 40];
    peak_width_limits= [0.5, 12];
    max_peaks        = 3;
    min_peak_height  = 0.3;    % in log10 ( ~3 dB)
    prox_threshold   = 2;
    aperiodic_mode   = 'fixed';  % or 'knee'

    % Prepare results struct array
    results(nElec) = struct('electrode_idx',[],'offset',[],'exponent',[],...
                            'knee',[],'n_peaks',[],'peak_params',[],...
                            'error',[],'r2',[],'BIC',[]);

    for e = 1:nElec
        logPxx = allPsds{e};
        [bestParams, ~] = ms_fooof_nll(allFreqs, logPxx, freqRange, ...
                               peak_width_limits, max_peaks, min_peak_height, ...
                               prox_threshold, aperiodic_mode);

        results(e).electrode_idx = e;
        if strcmpi(aperiodic_mode,'knee') && numel(bestParams.aperiodic_params)==3
            results(e).offset   = bestParams.aperiodic_params(1);
            results(e).knee     = bestParams.aperiodic_params(2);
            results(e).exponent = bestParams.aperiodic_params(3);
        else
            results(e).offset   = bestParams.aperiodic_params(1);
            results(e).exponent = bestParams.aperiodic_params(2);
            results(e).knee     = NaN;
        end

        pk = bestParams.peak_params;
        if any(pk(:)~=0)
            results(e).n_peaks     = size(pk,1);
            results(e).peak_params = pk;  % Nx3 [cfreq,amp,bw]
        else
            results(e).n_peaks     = 0;
            results(e).peak_params = [];
        end
        results(e).error = bestParams.MSE;
        results(e).r2    = bestParams.r_squared;
        results(e).BIC   = bestParams.BIC;
    end
    disp('Done fitting all electrodes with ms-specparam.');

    % -----------------------------------------------------------------
    % 5) SAVE RESULTS TO CSV
    % -----------------------------------------------------------------
    outCsv = fullfile(RESULTS_BASE, 'hup_ms_specparam_features.csv');
    fid = fopen(outCsv, 'w');
    fprintf(fid, 'electrode,offset,exponent,knee,n_peaks,MSE,r_squared,BIC,peak_params\n');

    for e = 1:nElec
        row = results(e);
        if row.n_peaks>0
            % each row of peak_params => [centerFreq amplitude bandwidth]
            pkStr = sprintf('[%.5f %.5f %.5f];', row.peak_params');
            pkStr = pkStr(1:end-1); % remove final semicolon
        else
            pkStr = '[]';
        end
        fprintf(fid, '%d,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,"%s"\n', ...
            row.electrode_idx, row.offset, row.exponent, row.knee, ...
            row.n_peaks, row.error, row.r2, row.BIC, pkStr);
    end
    fclose(fid);

    fprintf('Wrote ms-specparam results to: %s\n', outCsv);
    disp('All done. ms-FOOOF features in the CSV file.');
end


%% =================================================================
%                SUBFUNCTION: MS-SPECPARAM (NLL + BIC)
%% =================================================================
function [bestModel, allModels] = ms_fooof_nll(freqs, logPower, freqRange, ...
                peak_width_limits, max_peaks, min_peak_height, ...
                prox_threshold, aperiodic_mode)
% ms_fooof_nll:
%   Fit the log10(PSD) with a parametric 1/f (aperiodic) plus up to max_peaks,
%   evaluating negative log-likelihood & BIC. Return best model by BIC.

    % 1) restrict freq range
    fMask = (freqs>=freqRange(1) & freqs<=freqRange(2));
    xf = freqs(fMask);
    yf = logPower(fMask);

    % 2) Fit 0..max_peaks, pick best BIC
    allModels = [];
    for k = 0:max_peaks
        modelK = fit_one_model_k(xf, yf, k, aperiodic_mode, ...
                   peak_width_limits, min_peak_height, prox_threshold);
        allModels = [allModels modelK]; %#ok<AGROW>
    end

    allBICs = [allModels.BIC];
    [~, idx] = min(allBICs);
    bestModel = allModels(idx);
end


function M = fit_one_model_k(xf, yf, kPeaks, aperiodic_mode, ...
                             peak_width_limits, min_peak_height, proxThresh)
    % 1) initial robust aperiodic
    apInit = robust_ap_fit(xf, yf, aperiodic_mode);
    % 2) flatten
    flatSpec = yf - gen_aperiodic(xf, apInit, aperiodic_mode);
    % 3) find k guessed peaks
    if kPeaks>0
        guessPeaks = find_initial_peaks(xf, flatSpec, kPeaks, peak_width_limits, min_peak_height, proxThresh);
    else
        guessPeaks = zeros(0,3);
    end
    % 4) optimize all together
    finalParam = run_nll_optimize(xf, yf, aperiodic_mode, guessPeaks, peak_width_limits);

    % 5) evaluate final
    [MSE, r2, BIC, pkParams, apParams] = evaluate_model_fit(xf, yf, finalParam, aperiodic_mode);

    M.aperiodic_params = apParams;
    M.peak_params      = pkParams;
    M.MSE              = MSE;
    M.r_squared        = r2;
    M.BIC              = BIC;
end


%% ================== HELPER: aperiodic fits ==================
function y = gen_aperiodic(freqs, ap, mode)
    switch lower(mode)
        case 'fixed' % offset, exponent
            y = ap(1) - log10(freqs.^ap(2));
        case 'knee'  % offset, knee, exponent
            y = ap(1) - log10(abs(ap(2)) + freqs.^ap(3));
        otherwise
            error('unknown aperiodic mode');
    end
end

function apFinal = robust_ap_fit(freqs, pxx, mode)
    guess = simple_ap_fit(freqs, pxx, mode);
    iniFit= gen_aperiodic(freqs, guess, mode);
    r = pxx - iniFit;
    r(r<0)=0;
    th = prctile(r,2.5);
    mask = (r<=th);
    xf = freqs(mask);
    yf = pxx(mask);
    apFinal = simple_ap_fit(xf, yf, mode);
end

function ap = simple_ap_fit(freqs, pxx, mode)
    switch mode
        case 'fixed'
            exp_guess=-(pxx(end)-pxx(1))/log10(freqs(end)/freqs(1));
            guess=[pxx(1), exp_guess];
            opts=optimset('Display','off','TolFun',1e-6,'TolX',1e-6,'MaxIter',2000);
            ap=fminsearch(@(p) sum((pxx-(p(1)-log10(freqs.^p(2)))).^2), guess, opts);
        case 'knee'
            exp_guess=-(pxx(end)-pxx(1))/log10(freqs(end)/freqs(1));
            guess=[pxx(1),0,exp_guess];
            opts=optimset('Display','off','TolFun',1e-6,'TolX',1e-6,'MaxIter',2000);
            ap=fminsearch(@(p) sum((pxx-(p(1)-log10(abs(p(2))+freqs.^p(3)))).^2), guess, opts);
        otherwise
            error('unknown mode');
    end
end


%% ================== HELPER: peak guess ==================
function guessPeaks = find_initial_peaks(freqs, flatSpec, K, pwLimits, minPkHt, proxThresh)
    guessPeaks=zeros(K,3);
    specNow=flatSpec;
    df=freqs(2)-freqs(1);
    idxFill=1;
    for i=1:K
        [mxVal, idx] = max(specNow);
        if (mxVal<minPkHt)||(mxVal<1.0*std(specNow))
            break
        end
        cF = freqs(idx);
        halfht = 0.5*mxVal;
        leftI=idx; while (leftI>1 && specNow(leftI)>halfht), leftI=leftI-1; end
        rightI=idx;while (rightI<length(specNow) && specNow(rightI)>halfht), rightI=rightI+1; end
        fwhm=(rightI-leftI)*df;
        estSTD=fwhm/(2*sqrt(2*log(2)));
        if estSTD<pwLimits(1), estSTD=pwLimits(1); end
        if estSTD>pwLimits(2), estSTD=pwLimits(2); end
        guessPeaks(idxFill,:)=[cF, mxVal, estSTD];
        idxFill=idxFill+1;
        specNow=specNow - gaussian(freqs,cF,mxVal,estSTD);
    end
    guessPeaks(idxFill:end,:)=[];
    guessPeaks=drop_edge(guessPeaks,freqs(1),freqs(end),proxThresh);
    guessPeaks=drop_overlap(guessPeaks,proxThresh);
end

function y=gaussian(freqs, ctr, amp, sig)
    y=amp*exp(-0.5*((freqs-ctr)./sig).^2);
end

function g=drop_edge(g,fmin,fmax,thr)
    keep=true(size(g,1),1);
    for i=1:size(g,1)
        lo=g(i,1)-thr*g(i,3);
        hi=g(i,1)+thr*g(i,3);
        if (lo<fmin)||(hi>fmax), keep(i)=false; end
    end
    g=g(keep,:);
end

function g=drop_overlap(g, thr)
    if size(g,1)<2, return; end
    g=sortrows(g,1);
    i=1;
    while i<size(g,1)
        cF1=g(i,1); a1=g(i,2); s1=g(i,3);
        cF2=g(i+1,1);a2=g(i+1,2);s2=g(i+1,3);
        if (cF2-cF1)< thr*(s1+s2)
            % remove smaller amplitude
            if a1<a2
                g(i,:)=[];
            else
                g(i+1,:)=[];
            end
        else
            i=i+1;
        end
    end
end


%% ================== HELPER: main NLL optimization ==================
function bestParam=run_nll_optimize(freqs, pxx, mode, guessPeaks, pwLimits)
    if strcmpi(mode,'fixed'), nAp=2; else, nAp=3; end
    ap0 = simple_ap_fit(freqs, pxx, mode);
    x0  = [ap0(:); guessPeaks(:)'];

    lb=[]; ub=[];
    if strcmpi(mode,'fixed')
        lb=[-Inf; 0];
        ub=[ Inf; 10];
    else
        lb=[-Inf; 0; 0];
        ub=[ Inf; Inf; 10];
    end
    for i=1:size(guessPeaks,1)
        cF=guessPeaks(i,1);
        aM=guessPeaks(i,2);
        sD=guessPeaks(i,3);
        lb=[lb; max(0,cF-2*sD); 0; pwLimits(1)];
        ub=[ub; cF+2*sD; Inf; pwLimits(2)];
    end
    opts=optimset('Display','off','MaxFunEvals',5000,'MaxIter',5000,'TolFun',1e-9,'TolX',1e-7);
    try
        bestParam=fmincon(@(p) sum_of_squares_nll(p,freqs,pxx,mode), x0,[],[],[],[],lb,ub,[],opts);
    catch
        bestParam=x0;
    end
end

function val=sum_of_squares_nll(params,freqs,pxx,mode)
    model=build_model(freqs,params,mode);
    val=sum((pxx-model).^2);
end

function yhat=build_model(freqs, params, mode)
    if strcmpi(mode,'fixed')
        ap=params(1:2);
        pk=params(3:end);
    else
        ap=params(1:3);
        pk=params(4:end);
    end
    yhat=gen_aperiodic(freqs,ap,mode);
    for i=1:3:length(pk)
        cf=pk(i); amp=pk(i+1); bw=pk(i+2);
        yhat=yhat+gaussian(freqs,cf,amp,bw);
    end
end


%% ================== HELPER: finalize & compute BIC ==================
function [MSE, r2, BIC, pkOut, apOut] = evaluate_model_fit(freqs, pxx, paramVec, mode)
    if strcmpi(mode,'fixed')
        apOut=paramVec(1:2);
        pkOut=paramVec(3:end);
    else
        apOut=paramVec(1:3);
        pkOut=paramVec(4:end);
    end
    yfit=build_model(freqs,paramVec,mode);
    resid=pxx-yfit;
    MSE=mean(resid.^2);
    cc=corrcoef(pxx,yfit);
    if size(cc,1)>1
        r2=cc(1,2)^2;
    else
        r2=0;
    end
    np = length(paramVec);
    N  = length(freqs);
    logLik = -N/2*(1+log(MSE)+log(2*pi));
    BIC = np*log(N) - 2*logLik;

    if isempty(pkOut)
        pkOut=zeros(0,3);
    else
        pkOut=reshape(pkOut,3,[])';
    end
end
