12/05
TODO:
resolve names so that everything is clear
label in an extra file which functions are called within which main file


EFFECT SIZES:

1. Cohen's d


d = (M₁ - M₂) / s_pooled

where:
s_pooled = √[(s₁² + s₂²) / 2]
M₁, M₂ are means of the two groups
s₁, s₂ are standard deviations of the two groups

2. Mann-Whitney U Test:

Nonparametric alternative to t-test
Doesn't assume normality of distributions
Tests whether one group tends to have larger values than another
Works by ranking all observations and comparing rank sums between groups

Combine all values from both groups (HUP and MNI for each feature)
Rank all values from lowest (1) to highest (n1 + n2)

U1 = R1 - [n1(n1 + 1)/2]
where:
- R1 is sum of ranks for first group
- n1 is size of first group

The U Statistic represents:
Number of pairwise wins (how many times a value from group 1 is greater than a value from group 2)
Range: 0 to n1*n2
U/(n1*n2) gives probability that random value from group 1 is larger than random value from group 2

AUC = U/(n1*n2)
Range: 0 to 1
0.5 means no separation between groups
0.5 means group 1 tends to be larger
<0.5 means group 2 tends to be larger