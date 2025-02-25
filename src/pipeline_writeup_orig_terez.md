Spectral Feature Computation (per electrode):

Take raw time series data (EEG signal)
Apply Welch's method to compute power spectral density (PSD)

Using 2-second Hamming windows
50% overlap between windows
Frequency resolution determined by window length

Filter out line noise (57.5-62.5 Hz)
Compute band powers using trapezoidal integration:

delta (1-4 Hz)
theta (4-8 Hz)
alpha (8-13 Hz)
beta (13-30 Hz)
gamma (30-80 Hz)
broad (1-80 Hz)

Apply log transform to band powers
Calculate relative powers (normalized by total power)
Result: One set of features per electrode that summarizes its entire time series

Questions:
the paper says - we used first-order butterworth filter with a passband of 0.5 to 80Hz to remove HFOd
we also applied a 60Hz notch filter to remove line noise
To match the MNI atlas, we downsampled the data to 200Hz from its original 512 or 1024Hz sampling rate
Excluded all channels with a clear artifact as well as bipolar pairs in which both contacts were in white matter ,or either contact was outside the brain
"our method allows us to determine whether the distribution of power across frequency bands differs from the normtive distribution, but not whether absolute power is abnormally low/high"
"we aggregate all normative channels within a given region - calculate the power in each FB for each normative electrode and estimate the normal distrib across channels
comparison between the calculated relative band-power of the test channel with normative distribution
the process yields a z-score of spectral activity for each FB at each electrode contact in the test patient

After computing spectral features:
- what kind of statistical analysis?
- identifying outliers
- mapping to surgery outcome 

Other features
- coherence - is that bivariate?

Reproducing figures:
y-axis relative band power, concat with total power, x-axis all the frequency bands - paired test for the two cohorts, low effect size

Meeting 11/11:
are the normative maps different across centers?
we first need to make a site-specific normative map and then make comparisons across electrodes at the region level

First, assuming all of them are SF, we need to make a normative map that excludes resected tissue
that's the data that I gave you for MNI - that is already normative
HUP has all the data - but we have data to do the normative from it

questions: 
(1) are there differences between the site-specific datasets?
for that, 
(2) how much data do we need from each site to include in the map
(3) what is the effect of of atlas choice?

(1)
even before z-scores - when we calculate the Welch spectrum, 

## Background

## Motivation

## Objectives
**Are normative features between sites different?**  
1. calculate for each channel:
    Experiments:
    - (univariate) band power features:
        delta (1-4 Hz)
        theta (4-8 Hz)
        alpha (8-13 Hz)
        beta (13-30 Hz)
        gamma (30-80 Hz)
        broad (1-80 Hz)

    - (univariate) entropy-based features
    - (bivariate) coherence features:
        low priority
 

2. Group channels by atlas ROI (DK)
3. Check for global differences between sites across all ROIs
4. Check for differences between sites at individual ROI level
(none of these 1-4 require z-scores)

Expected results:
(i) Band power features do not have significant differences across sites when all ROIs are compared
(ii) Entropy features are differents across sites when all ROIs are compared
(iii) Some ROIs are more susceptible to differences than others

Tool: method to harmonize data while preserving the biological effects of age, sex, and epilepsy duration or other clinically relevant factors

**How much data do we need from each site to integrate that site in the normative atlas?**  

**What is the effect of atlas choice on harmonization?**

TODO
plot PSD on a log plot
take AUC to see total power
x-axis PSD, y-axis frequencies on the log scale
average PSD per each ROI - the shaded part is the confidence intervals across all electrodes
figure 4 
your fig - one fig for each ROI, each containing two curves - 
under Site Comparison point 3 2nd bulletpoint, for the first, reproduce first figure

Next steps:
from hup_iEEGnormal and mni -||- 
we have the feature for each electrode and we also have the coordinate of each electrode (ChannelPosition) -> you want to find nearest neighbor to assign coordinate

2 ways of doing 

NOTES ON STAT
Random effects: Patient, Region
Fixed effect: Site
Model: Feature ~ Site + (1|Patient) + (1|Region)
Account for nested data structure
Handle unbalanced sampling

STATISTICAL POWER AND EFFECT SIZES:
Electrode Count vs Effect Size:

Effect sizes measure the magnitude of difference between groups, not statistical power
Larger electrode counts increase confidence in the measurements but don't necessarily mean larger differences
Low effect sizes with high sampling suggest genuine similarity between sites in those regions

Statistical methodology:

Regions with <5 patients were filtered out (20 regions total)
Higher electrode counts provide more reliable estimates of regional activity
Multiple comparison correction (FDR) was applied
Mixed use of t-tests and Mann-Whitney U tests based on normality

Good balance of sensitivity (detecting real differences) and specificity (not finding differences where none exist)
Robust sampling in key regions enables confident interpretation
Regional differences emerge despite varying electrode counts, suggesting real physiological differences rather than sampling artifacts

Misc remarks:
Effect size:
Cohen's d compares two groups by expressing the difference between their means in standard deviation units. Computed by dividing the difference between two means by the data's standard deviation. Used to compare a treatment to a control group. A large Cohen's d indicates a large mean difference compared to the variability.

New features to add:
Autocorrelation: compares a time series to a lagged version of itself over multiple time intervals. It's similar to calculating the correlation between two different time series, but it uses the same time series twice. The autocorrelation function (ACF) measures how data points in a time series relate to the preceding data points. An autocorrelation of +1 indicates a perfect positive correlation, while an autocorrelation of -1 indicates a perfect negative correlation. 
A common way to detect autocorrelation is to plot the model residuals versus time.

Coherence:
The coherence is calculated by comparing the frequency content of the signals recorded at different sites on the scalp.
Coherence measures the synchronization between two signals based on phase consistency. A coherence value of 1 means that the two channels have the same phase difference, while a value close to 0 means the phase difference is random. 

DECEMBER
i'm pretty sure regional_analysis.py is outdated and can be deleted
